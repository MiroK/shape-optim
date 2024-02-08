from gmshnics.interopt import msh_gmsh_model, mesh_from_gmsh
import gmsh, sys


def channel_geometry(fl, bl, w, r, lc=0.1, R=None, center=None, view=False):
    '''
       fl+r      bl+r
    <------><------------->  ^
           x                 |
         circle center       | w
         (0, 0)              |
                             v
    '''
    gmsh.initialize(sys.argv)

    model = gmsh.model
    fac = model.occ
    
    A = fac.addPoint(-fl-r, -w/2, 0)
    B = fac.addPoint(r+bl, -w/2, 0)
    C = fac.addPoint(r+bl, w/2, 0)
    D = fac.addPoint(-fl-r, w/2, 0)

    bot = fac.addLine(A, B)
    right = fac.addLine(B, C)
    top = fac.addLine(C, D)
    left = fac.addLine(D, A)

    if center is None:
        center = (0, 0)
    if R is None:
        o = fac.addCircle(*center, 0, r)
    else:
        (r, R) = (R, r) if r < R else (r, R)
        o = fac.addEllipse(*center, 0, r, R)

    outer = fac.addCurveLoop([bot, right, top, left])
    inner = fac.addCurveLoop([o])
    surf = fac.addPlaneSurface([outer, inner])
    fac.synchronize()

    model.addPhysicalGroup(2, [surf], 1)
    model.addPhysicalGroup(1, [left], 1)
    model.addPhysicalGroup(1, [right], 2)
    model.addPhysicalGroup(1, [bot], 3)
    model.addPhysicalGroup(1, [top], 4)    
    model.addPhysicalGroup(1, [o], 5)    

    fac.synchronize()

    # Set sizes
    gmsh.model.mesh.field.add('Distance', 1)
    gmsh.model.mesh.field.setNumbers(1, 'CurvesList', [o])

    gmsh.model.mesh.field.add('Threshold', 2)
    gmsh.model.mesh.field.setNumber(2, 'InField', 1)
    gmsh.model.mesh.field.setNumber(2, 'SizeMin', lc / 5)
    gmsh.model.mesh.field.setNumber(2, 'SizeMax', lc)
    gmsh.model.mesh.field.setNumber(2, 'DistMin', 0.15)
    gmsh.model.mesh.field.setNumber(2, 'DistMax', 0.5)

    gmsh.model.mesh.field.setAsBackgroundMesh(2)
    
    if view:
        gmsh.fltk.initialize()
        gmsh.fltk.run()
    # Mesh it
    nodes, topologies = msh_gmsh_model(model, 2)
    mesh, entity_functions = mesh_from_gmsh(nodes, topologies)
    gmsh.finalize()
    
    return entity_functions[1]

# --------------------------------------------------------------------

if __name__ == '__main__':
    import dolfin as df
    import sys

    fl = 1
    bl = 1
    w = 1
    r = 0.2

    boundaries = channel_geometry(fl, bl, w, r, lc=0.1, R=None, center=None)
    
    # ----

    mesh = boundaries.mesh()
    # IO
    with df.XDMFFile(mesh.mpi_comm(), 'reference_domain_mesh.xdmf') as xdmf:
        xdmf.write(mesh, df.XDMFFile.Encoding.HDF5)

    with df.XDMFFile(mesh.mpi_comm(), 'reference_domain_boundaries.xdmf') as xdmf:
        xdmf.write(boundaries, df.XDMFFile.Encoding.HDF5)
        

