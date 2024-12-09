from docutils.nodes import Text, paragraph
from sphinx_automodapi.automodsumm import Automodsumm, process_automodsumm_generation


class AutoAPISummary(Automodsumm):
    def run(self):
        modprefix = self.arguments[0][self.arguments[0].rfind(".") + 1 :]
        nodes = super().run()
        for textnode in nodes[1].traverse(condition=Text, include_self=False):
            if not isinstance(textnode.parent, paragraph):
                newnode = Text(f"{modprefix}.{textnode.astext()}")
                textnode.parent.replace(textnode, newnode)

        return nodes


def setup(app):
    app.add_directive("automodsumm", AutoAPISummary)
    app.connect("builder-inited", process_automodsumm_generation)
