import copy

import numpy
import six

from chainer import cuda
from chainer import variable


class Link(object):

    """Building block of model definitions.

    Link is a building block of neural network models that support various
    features like handling parameters, defining network fragments,
    serialization, etc.

    Link is the primitive structure for the model definitions. It supports
    management of parameter variables and *persistent values* that should be
    incorporated to serialization. Parameters are variables registered via
    the :meth:`add_param` method, or given to the initializer method.
    Persistent values are arrays, scalars, or any other serializable values
    registered via the :meth:`add_persistent` method.

    .. note::
       Whereas arbitrary serializable objects can be registered as persistent
       values, it is strongly recommended to just register values that should
       be treated as results of learning. A typical example of persistent
       values is ones computed during training and required for testing, e.g.
       running statistics for batch normalization.

    Parameters and persistent values are referred by their names. They can be
    accessed as attributes of the links. Link class itself manages the lists
    of names of parameters and persistent values to distinguish parameters and
    persistent values from other attributes.

    Link can be composed into more complex models. This composition feature is
    supported by child classes like :class:`Chain` and :class:`ChainList`. One
    can create a chain by combining one or more links. See the documents for
    these classes for details.

    As noted above, Link supports the serialization protocol of the
    :class:`~chainer.Serializer` class. **Note that only parameters and
    persistent values are saved and loaded.** Other attributes are considered
    as a part of user program (i.e. a part of network definition). In order to
    construct a link from saved file, other attributes must be identically
    reconstructed by user codes.

    .. admonition:: Example

       This is a simple example of custom link definition. Chainer itself also
       provides many links defined under the :mod:`~chainer.links` module. They
       might serve as examples, too.

       Consider we want to define a simple primitive link that implements a
       fully-connected layer based on the :func:`~functions.linear` function.
       Note that this function takes input units, a weight variable, and a bias
       variable as arguments. Then, the fully-connected layer can be defined as
       follows::

          import chainer
          import chainer.functions as F
          import numpy as np

          class LinearLayer(chainer.Link):

              def __init__(self, n_in, n_out):
                  # Parameters are initialized as a numpy array of given shape.
                  super(LinearLayer, self).__init__(
                      W=(n_out, n_in),
                      b=(n_out,),
                  )
                  self.W.data[...] = np.random.randn(n_out, n_in)
                  self.b.data.fill(0)

              def __call__(self, x):
                  return F.linear(x, self.W, self.b)

       This example shows that a user can define arbitrary parameters and use
       them in any methods. Links typically implement the ``__call__``
       operator.

    Args:
        params: Shapes of initial parameters. The keywords are used as their
            names. The names are also set to the parameter variables.

    Attributes:
        name (str): Name of this link, given by the parent chain (if exists).

    """
    def __init__(self, **params):
        self._params = []
        self._persistent = []
        self._cpu = True
        self.name = None

        for name, shape in six.iteritems(params):
            self.add_param(name, shape)

    @property
    def xp(self):
        """Array module for this link.

        Depending on which of CPU/GPU this link is on, this property returns
        :mod:`numpy` or :mod:`cupy`.

        """
        return numpy if self._cpu else cuda.cupy

    def add_param(self, name, shape, dtype=numpy.float32):
        """Registers a parameter to the link.

        The registered parameter is saved and loaded on serialization and
        deserialization, and involved in the optimization. The data and
        gradient of the variable are initialized by NaN arrays.

        The parameter is set to an attribute of the link with the given name.

        Args:
            name (str): Name of the parameter. This name is also used as the
                attribute name.
            shape (int or tuple of ints): Shape of the parameter array.
            dtype: Data type of the parameter array.

        """
        d = self.__dict__
        if name in d:
            raise AttributeError(
                'cannot register a new parameter %s: attribute exists'
                % name)
        data = self.xp.full(shape, numpy.nan, dtype=dtype)
        grad = data.copy()
        var = variable.Variable(data, volatile='auto', name=name)
        var.grad = grad
        self._params.append(name)
        d[name] = var

    def add_persistent(self, name, value):
        """Registers a persistent value to the link.

        The registered value is saved and loaded on serialization and
        deserialization. The value is set to an attribute of the link.

        Args:
            name (str): Name of the persistent value. This name is also used
                for the attribute name.
            value: Value to be registered.

        """
        d = self.__dict__
        if name in d:
            raise AttributeError(
                'cannot register a new persistent value %s: attribute exists'
                % name)
        self._persistent.append(name)
        d[name] = value

    def copy(self):
        """Copies the link hierarchy to new one.

        The whole hierarchy rooted by this link is copied. The copy is
        basically shallow, except that the parameter variables are also
        shallowly copied. It means that the parameter variables of copied one
        are different from ones of original link, while they share the data and
        gradient arrays.

        The name of the link is reset on the copy, since the copied instance
        does not belong to the original parent chain (even if exists).

        Returns:
            Link: Copied link object.

        """
        ret = copy.copy(self)
        ret._params = list(self._params)
        ret._persistent = list(self._persistent)
        ret.name = None
        d = ret.__dict__
        for name in ret._params:
            d[name] = copy.copy(d[name])
        return ret

    def to_cpu(self):
        """Copies parameter variables and persistent values to CPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to CPU, the link implementation must
        override this method to do so.

        Returns: self

        """
        if self._cpu:
            return self
        d = self.__dict__
        for name in self._params:
            d[name].to_cpu()
        for name in self._persistent:
            value = d[name]
            if isinstance(value, cuda.ndarray):
                d[name] = value.get()
        self._cpu = True
        return self

    def to_gpu(self, device=None):
        """Copies parameter variables and persistent values to GPU.

        This method does not handle non-registered attributes. If some of such
        attributes must be copied to GPU, the link implementation must
        override this method to do so.

        Args:
            device: Target device specifier. If omitted, the current device is
                used.

        Returns: self

        """
        cuda.check_cuda_available()
        if not self._cpu:
            return self
        d = self.__dict__
        with cuda.get_device(device):
            for name in self._params:
                d[name].to_gpu()
            for name in self._persistent:
                value = d[name]
                if isinstance(value, numpy.ndarray):
                    d[name] = cuda.to_gpu(value)
        self._cpu = False
        return self

    def params(self):
        """Returns a generator of all parameters under the link hierarchy.

        Returns:
            A generator object that generates all parameters.

        """
        d = self.__dict__
        for name in self._params:
            yield d[name]

    def namedparams(self):
        """Returns a generator of all (path, param) pairs under the hierarchy.

        Returns:
            A generator object that generates all (path, parameter) pairs. The
            paths are relative from this link.

        """
        d = self.__dict__
        for name in self._params:
            yield '/' + name, d[name]

    def links(self, skipself=False):
        """Returns a generator of all links under the hierarchy.

        Args:
            skipself (bool): If ``True``, then the generator skips this link
                and starts with the first child link.

        Returns:
            A generator object that generates all links.

        """
        if not skipself:
            yield self

    def namedlinks(self, skipself=False):
        """Returns a generator of all (path, link) pairs under the hierarchy.

        Args:
            skipself (bool): If ``True``, then the generator skips this link
                and starts with the first child link.

        Returns:
            A generator object that generates all (path, link) pairs.

        """
        if not skipself:
            yield '/', self

    def children(self):
        """Returns a generator of all child links.

        Returns:
            A generator object that generates all child links.

        """
        if 0:
            yield

    def copyparams(self, link):
        """Copies all parameters from given link.

        This method copies data arrays of all parameters in the hierarchy. The
        copy is even done across the host and devices. Note that this method
        does not copy the gradient arrays.

        Args:
            link (Link): Source link object.

        """
        src = link.__dict__
        dst = self.__dict__
        for name in self._params:
            dst[name].copydata(src[name])

    def zerograds(self):
        """Initializes all gradient arrays by zero.

        This method should be called before the backward computation at every
        iteration of the optimization.

        """
        for param in self.params():
            param.zerograd()

    def addgrads(self, link):
        """Accumulates gradient values from given link.

        This method adds each gradient array of the given link to corresponding
        gradient array of this link. The accumulation is even done across
        host and different devices.

        Args:
            link (Link): Source link object.

        """
        src = link.__dict__
        dst = self.__dict__
        for name in self._params:
            dst[name].addgrad(src[name])

    def serialize(self, serializer):
        """Serializes the link object.

        Args:
            serializer (~chainer.AbstractSerializer): Serializer object.

        """
        d = self.__dict__
        for name in self._params:
            serializer(name, d[name].data)
        for name in self._persistent:
            d[name] = serializer(name, d[name])


class Chain(Link):

    """Composable link with object-like interface.

    Composability is one of the most important features of neural nets. Neural
    net models consist of many reusable fragments, and each model itself might
    be embedded into a larger learnable system. Chain enables us to write a
    neural net based on composition, without bothering about routine works like
    collecting parameters, serialization, copying the structure with parameters
    shared, etc.

    This class actually provides a way to compose one or more links into one
    structure. A chain can contain one or more *child links*. Child link is a
    link registered to the chain with its own name. The child link is stored to
    an attribute of the chain with the name. User can write a whole model or a
    fragment of neural nets as a child class of Chain.

    Each chain itself is also a link. Therefore, one can combine chains into
    higher-level chains. In this way, links and chains construct a *link
    hierarchy*. Link hierarchy forms a tree structure, where each node is
    identified by the path from the root. The path is represented by a string
    like a file path in UNIX, consisting of names of nodes on the path, joined
    by slashes ``/``.

    .. admonition:: Example

       This is a simple example of custom chain definition. Chainer itself also
       provides some chains defined under the :mod:`~chainer.links` module.
       They might serve as examples, too.

       Consider we want to define a multi-layer perceptron consisting of two
       hidden layers with rectifiers as activation functions. We can use the
       :class:`~chainer.links.Linear` link as a building block::

          import chainer
          import chainer.functions as F
          import chainer.links as L

          class MultiLayerPerceptron(chainer.Chain):

              def __init__(self, n_in, n_hidden, n_out):
                  # Create and register three layers for this MLP
                  super(MultiLayerPerceptron, self).__init__(
                      layer1=L.Linear(n_in, n_hidden),
                      layer2=L.Linear(n_hidden, n_hidden),
                      layer3=L.Linear(n_hidden, n_out),
                  )

              def __call__(self, x):
                  # Forward propagation
                  h1 = F.relu(self.layer1(x))
                  h2 = F.relu(self.layer2(h1))
                  return self.layer3(h2)

       Child links are registered via the initializer method. They also can be
       registered by the :meth:`add_link` method. The forward propagation is
       often implemented as The ``__call__`` operator as the above example,
       though it is not mandatory.

    Args:
        links: Child links. The keywords are used as their names. The names are
            also set to the links.

    """
    def __init__(self, **links):
        super(Chain, self).__init__()
        self._children = []

        for name, link in six.iteritems(links):
            self.add_link(name, link)

    def __getitem__(self, name):
        """Equivalent to getattr."""
        return getattr(self, name)

    def add_link(self, name, link):
        """Registers a child link to this chain.

        The registered link is saved and loaded on serialization and
        deserialization, and involved in the optimization. The registered link
        is called a child. The child link is set to an attribute of the chain
        with the given name.

        This method also sets the :attr:`~Link.name` attribute of the
        registered link. If the given link already has the name attribute set,
        then it raises an error.

        Args:
            name (str): Name of the child link. This name is also used as the
                attribute name.
            link (Link): The link object to be registered.

        """
        if link.name is not None:
            raise ValueError(
                'given link is already registered to another chain by name %s'
                % link.name)
        d = self.__dict__
        if name in d:
            raise AttributeError(
                'cannot register a new link %s: attribute exists' % name)
        self._children.append(name)
        link.name = name
        d[name] = link

    def copy(self):
        ret = super(Chain, self).copy()
        ret._children = list(ret._children)
        d = ret.__dict__
        for name in ret._children:
            # copy child links recursively
            copied = d[name].copy()
            copied.name = name
            d[name] = copied
        return ret

    def to_cpu(self):
        super(Chain, self).to_cpu()
        d = self.__dict__
        for name in self._children:
            d[name].to_cpu()
        return self

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            super(Chain, self).to_gpu()
            d = self.__dict__
            for name in self._children:
                d[name].to_gpu()
        return self

    def params(self):
        for param in super(Chain, self).params():
            yield param
        d = self.__dict__
        for name in self._children:
            for param in d[name].params():
                yield param

    def namedparams(self):
        for ret in super(Chain, self).namedparams():
            yield ret
        d = self.__dict__
        for name in self._children:
            prefix = '/' + name
            for path, param in d[name].namedparams():
                yield prefix + path, param

    def links(self, skipself=False):
        if not skipself:
            yield self
        d = self.__dict__
        for name in self._children:
            for link in d[name].links():
                yield link

    def namedlinks(self, skipself=False):
        if not skipself:
            yield '/', self
        d = self.__dict__
        for name in self._children:
            child = d[name]
            prefix = '/' + name
            yield prefix, child
            for path, link in d[name].namedlinks(True):
                yield prefix + path, link

    def children(self):
        d = self.__dict__
        for name in self._children:
            yield d[name]

    def copyparams(self, link):
        super(Chain, self).copyparams(link)
        src = link.__dict__
        dst = self.__dict__
        for name in self._children:
            dst[name].copyparams(src[name])

    def zerograds(self):
        super(Chain, self).zerograds()
        d = self.__dict__
        for name in self._children:
            d[name].zerograds()

    def addgrads(self, link):
        super(Chain, self).addgrads(link)
        src = link.__dict__
        dst = self.__dict__
        for name in self._children:
            dst[name].addgrads(src[name])

    def serialize(self, serializer):
        super(Chain, self).serialize(serializer)
        d = self.__dict__
        for name in self._children:
            d[name].serialize(serializer[name])


class ChainList(Link):

    """Composable link with list-like interface.

    This is another example of compositional link. Unlike :class:`Chain`, this
    class can be used like a list of child links. Each child link is indexed by
    a non-negative integer, and it maintains the current number of registered
    child links. The :meth:`add_link` method inserts a new link at the end of
    the list. It is useful to write a chain with arbitrary number of child
    links, e.g. an arbitrarily deep multi-layer perceptron.

    Note that this class does not implement all methods of :class:`list`.

    Args:
        links: Initial child links.

    """
    def __init__(self, *links):
        super(ChainList, self).__init__()
        self._children = []

        for link in links:
            self.add_link(link)

    def __getitem__(self, index):
        """Returns the child at given index.

        Args:
            index (int): Index of the child in the list.

        Returns:
            Link: The ``index``-th child link.

        """
        return self._children[index]

    def __iter__(self):
        return iter(self._children)

    def __len__(self):
        """Returns a number of children."""
        return len(self._children)

    def add_link(self, link):
        """Registers a child link to this chain.

        The registered link is saved and loaded on serialization and
        deserialization, and involved in the optimization. The registered link
        is called a child. The child link is accessible via :meth:`children`
        generator, which returns a generator running through the children in
        registered order.

        This method also sets the :attr:`~Link.name` attribute of the
        registered link. If the given link already has the name attribute set,
        then it raises an error.

        Args:
            link (Link): The link object to be registered.

        """
        if link.name is not None:
            raise ValueError(
                'given link is already registered to another chain by name %s'
                % link.name)
        link.name = str(len(self._children))
        self._children.append(link)

    def copy(self):
        ret = super(ChainList, self).copy()
        ret._children = list(ret._children)  # copy
        children = ret._children
        for i, child in enumerate(children):
            child = child.copy()
            child.name = str(i)
            children[i] = child
        return ret

    def to_cpu(self):
        super(ChainList, self).to_cpu()
        for link in self._children:
            link.to_cpu()
        return self

    def to_gpu(self, device=None):
        with cuda.get_device(device):
            super(ChainList, self).to_gpu()
            for link in self._children:
                link.to_gpu()
        return self

    def params(self):
        for param in super(ChainList, self).params():
            yield param
        for link in self._children:
            for param in link.params():
                yield param

    def namedparams(self):
        for ret in super(ChainList, self).namedparams():
            yield ret
        for idx, link in enumerate(self._children):
            prefix = '/%d' % idx
            for path, param in link.namedparams():
                yield prefix + path, param

    def links(self, skipself=False):
        if not skipself:
            yield self
        for child in self._children:
            for link in child.links():
                yield link

    def namedlinks(self, skipself=False):
        if not skipself:
            yield '/', self
        for idx, child in enumerate(self._children):
            prefix = '/%d' % idx
            yield prefix, child
            for path, link in child.namedlinks(True):
                yield prefix + path, link

    def children(self):
        for child in self._children:
            yield child

    def copyparams(self, link):
        super(ChainList, self).copyparams(link)
        for idx, child in enumerate(self._children):
            child.copyparams(link[idx])

    def zerograds(self):
        super(ChainList, self).zerograds()
        for child in self._children:
            child.zerograds()

    def addgrads(self, link):
        super(ChainList, self).addgrads(link)
        for idx, child in enumerate(self._children):
            child.addgrads(link[idx])

    def serialize(self, serializer):
        super(ChainList, self).serialize(serializer)
        for idx, child in enumerate(self._children):
            child.serialize(serializer['%d' % idx])
