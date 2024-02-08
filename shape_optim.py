# Attempt on shape optim with everything formulated in the reference domain
# Mesh deformation uses Laplacian
from mesh_generation import channel_geometry
from dolfin import *
from dolfin_adjoint import *
import numpy as np


def harmonic_extension_neumann(displacement, facet_f, g=None):
    '''Using control for Neuamann bcs'''
    mesh = displacement.function_space().mesh()
    
    # Displacement will be defined via control ...
    if g is None:
        G = FunctionSpace(mesh, 'DG', 0)
        g = Function(G)

    n = FacetNormal(mesh)

    d, dd = TrialFunction(D), TestFunction(D)
    a_harm = inner(grad(d), grad(dd))*dx
    # ... as a Neumann bcs in this case
    L_harm = inner(Constant((0, 0)), dd)*dx + inner(g, dot(dd, n))*ds(5)

    harm_bcs = [DirichletBC(D, Constant((0, 0)), facet_f, 1),
                DirichletBC(D, Constant((0, 0)), facet_f, 2),
                DirichletBC(D, Constant((0, 0)), facet_f, 3),
                DirichletBC(D, Constant((0, 0)), facet_f, 4)]

    A_harm, b_harm = assemble_system(a_harm, L_harm, harm_bcs)
    solve(A_harm, displacement.vector(), b_harm)

    return g

# --------------------------------------------------------------------

if __name__ == '__main__':

    fl = 1
    bl = 1
    w = 1
    r = 0.2

    boundaries = channel_geometry(fl, bl, w, r, lc=0.1, R=None, center=None)

    mesh = boundaries.mesh()
    # IO
    with XDMFFile(mesh.mpi_comm(), 'reference_domain_mesh.xdmf') as xdmf:
        xdmf.write(mesh, XDMFFile.Encoding.HDF5)

    with XDMFFile(mesh.mpi_comm(), 'reference_domain_boundaries.xdmf') as xdmf:
        xdmf.write(boundaries, XDMFFile.Encoding.HDF5)

    # ---------

    mesh = Mesh()
    with XDMFFile(MPI.comm_world, 'reference_domain_mesh.xdmf') as mesh_xdmf:
        mesh_xdmf.read(mesh)

    facet_f = MeshFunction('size_t', mesh, mesh.topology().dim() - 1)
    with XDMFFile(MPI.comm_world, 'reference_domain_boundaries.xdmf') as mesh_xdmf:
        mesh_xdmf.read(facet_f)
    ds = ds(domain=mesh, subdomain_data=facet_f)

    # -----

    V_h = VectorElement('CG', mesh.ufl_cell(), 2)
    Q_h = FiniteElement('CG', mesh.ufl_cell(), 1)
    W = FunctionSpace(mesh, V_h * Q_h)
    V, Q = W.split()

    v, q = TestFunctions(W)
    x = TrialFunction(W)
    u, p = split(x)
    s = Function(W, name='Stokes')
    V_collapse = V.collapse()                  # Will use this space for displacement too

    D = VectorFunctionSpace(mesh, 'CG', 1)
    dh = Function(D, name='Displacement')
    # Define the problem for displacement dh returning a control variable
    g = harmonic_extension_neumann(dh, facet_f)

    # With displacement define deformation and transform operators
    x = SpatialCoordinate(mesh)

    fmap = x + dh
    F = grad(fmap)
    J = det(F)

    Grad = lambda arg: dot(grad(arg), inv(F))
    Div = lambda arg: tr(Grad(arg))

    # Now we can define Stokes on deformed mesh
    mu = Constant(1)

    a = (mu*inner(sym(Grad(u)), sym(Grad(v)))*J*dx - inner(p, Div(v))*J*dx
         - inner(q, Div(u))*J*dx)

    L = inner(Constant((0, 0)), v)*J*dx # So the outflow pressure is set to 0

    # Define boundary conditions
    u_inflow = Expression(('-(x[1]-0.5)*(x[1]+0.5)', '0'), degree=2)

    inflow = [DirichletBC(W.sub(0), u_inflow, facet_f, 1)]

    wall = [DirichletBC(W.sub(0), Constant((0, 0)), facet_f, 3),
            DirichletBC(W.sub(0), Constant((0, 0)), facet_f, 4),
            DirichletBC(W.sub(0), Constant((0, 0)), facet_f, 5)]

    # On outflow I want to constraint the tangent component
    outflow = [DirichletBC(W.sub(0).sub(1), Constant(0), facet_f, 2)]

    bcs = inflow + wall + outflow

    A, b = assemble_system(a, L, bcs)
    solve(A, s.vector(), b)

    # Create the reduced functional.
    u, p = split(s)

    dissipation = assemble((mu/2)*inner(sym(Grad(u)), sym(Grad(u)))*J*dx)

    # For the volume constraint, the hole should keep its volume
    alpha_vol = 10_000

    xmin, xmax = np.min(mesh.coordinates(), axis=0), np.max(mesh.coordinates(), axis=0)
    length, width = xmax - xmin
    full_area = 0 #length*width

    ref_domain_area = assemble(Constant(1)*dx(domain=mesh))
    hole_area = full_area - ref_domain_area

    current_hole_area = full_area - assemble(J*dx)

    volume_cstr = alpha_vol*(current_hole_area - hole_area)**2

    # Some regularizion of the bdry?
    alpha_sur = 1E-1
    reg = alpha_sur*assemble(inner(g, g)*ds(5))

    # Final loss
    loss = dissipation + volume_cstr + reg

    loss_reduced = ReducedFunctional(loss, Control(g))

    g_opt = minimize(loss_reduced, options={'disp': True}, method='L-BFGS-B')

    g.assign(g_opt)
    harmonic_extension_neumann(dh, facet_f, g=g)    

    A, b = assemble_system(a, L, bcs)
    solve(A, s.vector(), b)

    # ---
    File('control.pvd') << g

    ALE.move(mesh, dh)

    uh, ph = s.split(deepcopy=True)
    File('uh_optim.pvd') << uh
    File('ph_optim.pvd') << ph
