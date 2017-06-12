{{ fullname }}
{{ underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   ..
      Methods

{% block methods %}

   .. rubric:: Methods

   ..
      Special methods

{% for item in ('__call__', '__enter__', '__exit__', '__getitem__', '__setitem__', '__len__', '__next__', '__iter__', '__copy__') %}
{% if item in all_methods %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}

   ..
      Ordinary methods

{% for item in methods %}
{% if item not in ('__init__',) %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}

   ..
      Special methods

{% for item in ('__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__', '__nonzero__', '__bool__') %}
{% if item in all_methods %}
   .. automethod:: {{ item }}
{% endif %}
{%- endfor %}
{% endblock %}

   ..
      Atributes

{% block attributes %} {% if attributes %}

   .. rubric:: Attributes

{% for item in attributes %}
   .. autoattribute:: {{ item }}
{%- endfor %}
{% endif %} {% endblock %}
