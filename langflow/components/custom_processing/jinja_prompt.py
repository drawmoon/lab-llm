"""This is an implementation example of a custom component. Its main function is to implement a prompt template that supports jinja2.

This module provides a custom component that extends Langflow's functionality by implementing
a prompt template that supports jinja2 templating. It allows for more flexible and powerful
template rendering capabilities compared to standard string formatting.
"""

from langflow.custom.custom_component.component import Component


class JinjaPromptComponent(Component):
    name = "JinjaPromptTemplate"
    display_name = "Jinja Prompt Template"
