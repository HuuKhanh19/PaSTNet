"""Full PaST with m_full used in place of m_full - m_self at every stage/order."""

from pastnet.model.network import PaSTNet
from pastnet.model.past import PaSTBlock


class PaSTNetNoSingletonSubtraction(PaSTNet):
    def __init__(self, **options):
        super().__init__(**options)
        blocks = [module for module in self.modules() if isinstance(module, PaSTBlock)]
        if len(blocks) != 12:
            raise ValueError("Expected three PaST stages and four path orders")
        for block in blocks:
            block.singleton_subtraction = False
