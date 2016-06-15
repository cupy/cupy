Comparison with Other Frameworks
================================

A table for quick comparison
----------------------------

This table compares Chainer with other popular deep learning frameworks.
We hope it helps you to choose an appropriate framework for the demand.

.. note::

   This chart may be out-dated, since the developers of Chainer do not perfectly follow the latest development status of each framework.
   Please report us if you find an out-dated cell.
   Requests for new comparison axes are also welcome.


+-------+-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       |                             | Chainer           | Theano-based           | Torch7              | Caffe                                              |
+=======+=============================+===================+========================+=====================+====================================================+
| Specs | Scripting                   | Python            | Python                 | LuaJIT              | Python                                             |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | Net definition language     | Python            | Python                 | LuaJIT              | Protocol Buffers                                   |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | Define-by-Run scheme        | Y                 |                        |                     |                                                    |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | CPU Array backend           | NumPy             | NumPy                  | Tensor              |                                                    |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | GPU Array backend           | CuPy              | CudaNdarray [1]_       | CudaTensor          |                                                    |
+-------+-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
| NNs   | Reverse-mode AD             | Y                 | Y                      | Y                   | Y                                                  |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | Basic RNN support           | Y                 | Y                      | Y (``nnx``)         | `#2033 <https://github.com/BVLC/caffe/pull/2033>`_ |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | Variable-length loops       | Y                 | Y (``scan``)           |                     |                                                    |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | Stateful RNNs [2]_          | Y                 | Y                      | Y [6]_              |                                                    |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | Per-batch architectures     | Y                 |                        |                     |                                                    |
+-------+-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
| Perf  | CUDA support                | Y                 | Y                      | Y                   | Y                                                  |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | cuDNN support               | Y                 | Y                      | Y (``cudnn.torch``) | Y                                                  |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | FFT-based convolution       |                   | Y                      | Y (``fbcunn``)      | `#544 <https://github.com/BVLC/caffe/pull/544>`_   |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | CPU/GPU generic coding [3]_ | Y                 | [4]_                   | Y                   |                                                    |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | Multi GPU (data parallel)   | Y                 | Y [7]_                 | Y (``fbcunn``)      | Y                                                  |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | Multi GPU (model parallel)  | Y                 | Y [8]_                 | Y (``fbcunn``)      |                                                    |
+-------+-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
| Misc  | Type checking               | Y                 | Y                      | Y                   | N/A                                                |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | Model serialization         | Y                 | Y (``pickle``)         | Y                   | Y                                                  |
|       +-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+
|       | Caffe reference model       | Y                 | [5]_                   | Y (``loadcaffe``)   | Y                                                  |
+-------+-----------------------------+-------------------+------------------------+---------------------+----------------------------------------------------+

.. [1] They are also developing `libgpuarray <http://deeplearning.net/software/libgpuarray/>`_
.. [2] Stateful RNN is a type of RNN implementation that maintains states in the loops. It should enable us to use the states arbitrarily to update them.
.. [3] This row shows whether each array API supports unified codes for CPU and GPU.
.. [4] The array backend of Theano does not have compatible interface with NumPy, though most users write code on Theano variables, which is generic for CPU and GPU.
.. [5] Depending on the frameworks.
.. [6] Also available in the `Torch RNN package <https://github.com/Element-Research/rnn>`_
.. [7] Via `Platoon <https://github.com/mila-udem/platoon/>`_
.. [8] `Experimental as May 2016 <http://deeplearning.net/software/theano/tutorial/using_multi_gpu.html>`_

Benchmarks
----------

We are preparing for the benchmarks.
