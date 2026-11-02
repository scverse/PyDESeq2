{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:

   {% block methods %}
   {% set own = methods | reject("in", inherited_members) | reject("eq", "__init__") | list %}
   {% if own %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in own %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set own = attributes | reject("in", inherited_members) | list %}
   {% if own %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in own %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
