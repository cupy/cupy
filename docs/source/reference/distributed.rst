----------------
Distributed
----------------

Communication between processes
-------------------------------

.. module:: cupyx.distributed

.. autosummary::
   :toctree: generated/

   init_process_group
   NCCLBackend

``ndarray`` distributed across devices
--------------------------------------

.. module:: cupyx.distributed.array

.. autosummary::
   :toctree: generated/

   distributed_array
   DistributedArray
   make_2d_index_map
   matmul
