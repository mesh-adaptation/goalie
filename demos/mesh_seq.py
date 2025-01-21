# Creating a mesh sequence
# ========================
#
# In the `previous demo <./time_partition.py.html>`__,
# we saw how to create a :class:`TimePartition` instance
# - one of the fundamental objects in Goalie. Another
# fundamental object is the mesh sequence, :class:`MeshSeq`,
# which corresponds to a time partition. The idea is
# that a single mesh is associated with each subinterval.
#
# For this and subsequent demos, we import from the namespaces
# of both Firedrake and Goalie. ::

from firedrake import *

from goalie import *

# Again, turn debugging mode on to get verbose output. ::

set_log_level(DEBUG)

# We use Firedrake's utility :func:`UnitSquareMesh` function
# to create a list of two meshes with different resolutions. ::

m, n = 32, 16
meshes = [UnitSquareMesh(m, m), UnitSquareMesh(n, n)]

# Creating a :class:`MeshSeq` is as simple as ::

mesh_seq = MeshSeq(meshes)

# With debugging turned on, we get a report of the number of
# elements and vertices in each mesh in the sequence, as well
# as the corresponding maximal aspect ratio over all elements.
#
# .. code-block:: console
#
#     DEBUG MeshSeq: 0:    2048 cells,    1089 vertices,  max aspect ratio 1.21
#     DEBUG MeshSeq: 1:     512 cells,     289 vertices,  max aspect ratio 1.21
#
# We can plot the meshes comprising the :class:`MeshSeq` using
# the :meth:`plot` method, provided they are two dimensional. ::

fig, axes = mesh_seq.plot()
fig.savefig("mesh_seq.jpg")

# .. figure:: mesh_seq.jpg
#    :figwidth: 90%
#    :align: center

# In the `next demo <./ode.py.html>`__, we solve an ordinary differential equation (ODE)
# using a special kind of :class:`MeshSeq`.
#
# This demo can also be accessed as a `Python script <mesh_seq.py>`__.
