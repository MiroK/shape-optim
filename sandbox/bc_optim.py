# This is a slight modification of dolfin-adjoin demo
# http://www.dolfin-adjoint.org/en/stable/documentation/stokes-bc-control/stokes-bc-control.html

from dolfin import *
from dolfin_adjoint import *


mesh = Mesh()

with XDMFFile(MPI.comm_world, 'reference_domain_mesh.xdmf') as mesh_xdmf:
    mesh_xdmf.read(mesh)

V_h = VectorElement("CG", mesh.ufl_cell(), 2)
Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, V_h * Q_h)
V, Q = W.split()

v, q = TestFunctions(W)
x = TrialFunction(W)
u, p = split(x)
s = Function(W, name="State")
V_collapse = V.collapse()
g = Function(V_collapse, name="Control")

nu = Constant(1)

facet_f = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
with XDMFFile(MPI.comm_world, 'reference_domain_boundaries.xdmf') as mesh_xdmf:
    mesh_xdmf.read(facet_f)


ds = ds(domain=mesh, subdomain_data=facet_f)

# Define boundary conditions
u_inflow = Expression(("-(x[1]-0.5)*(x[1]+0.5)", "0"), degree=2)

inflow = [DirichletBC(W.sub(0), u_inflow, facet_f, 1)]

wall = [DirichletBC(W.sub(0), Constant((0, 0)), facet_f, 3),
        DirichletBC(W.sub(0), Constant((0, 0)), facet_f, 4)]

circle = [DirichletBC(W.sub(0), g, facet_f, 5)]

# On outflow I want to constraint the tangent component
outflow = [DirichletBC(W.sub(0).sub(1), Constant(0), facet_f, 2)]

bcs = inflow + wall + circle + outflow

a = (nu*inner(sym(grad(u)), sym(grad(v)))*dx - inner(p, div(v))*dx
     - inner(q, div(u))*dx)

n = FacetNormal(mesh)
p0 = Constant(0)
L = inner(Constant((0, 0)), v)*dx + inner(p0, dot(v, n))*ds(2)

A, b = assemble_system(a, L, bcs)
solve(A, s.vector(), b)

# Create the reduced functional.
u, p = split(s)
alpha = Constant(1E-5)

J = assemble(1./2*inner(grad(u), grad(u))*dx + alpha/2*inner(g, g)*ds(5))
m = Control(g)
Jhat = ReducedFunctional(J, m)

g_opt = minimize(Jhat, options={"disp": True})
plot(g_opt, title="Optimised boundary")

g.assign(g_opt)
A, b = assemble_system(a, L, bcs)
solve(A, s.vector(), b)

uh, ph = s.split(deepcopy=True)
File('uh_optim.pvd') << uh
File('ph_optim.pvd') << ph
print('|s|', s.vector().norm('l2'))
