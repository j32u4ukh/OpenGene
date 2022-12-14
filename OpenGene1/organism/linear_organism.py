import numpy as np

from OpenGene1.organism import Organism
from OpenGene1.structure.linear_structure import LinearStructure


# 直線結構型生物: 採用"直線串接型"的生物，不會因任務不同而個別設立類別
class LinearOrganism(Organism):
    def __init__(self, gene: np.array, n_cell: int = -1, energy: float = 100.0, mutation_rate: float = 0.03):
        super().__init__(gene=gene, n_cell=n_cell, energy=energy, mutation_rate=mutation_rate)

    def buildStructure(self):
        self.structure = LinearStructure(gene=self.structure_gene, n_node=self.n_cell)

    def buildOrganism(self):
        self.structure.loadCells(cells=self.cells)

    def call(self):
        pass

    def toParams(self):
        pass


if __name__ == "__main__":
    # n_cell, gene = Organism.createGene()
    # ulo = LinearOrganism(gene=gene, n_cell=n_cell, energy=100, mutation_rate=0.03)
    gene = Organism.manualSettingGene(base3=1)
    print(f"number_gene: {gene[:8]}")
    print(f"structure_gene: {gene[8:17]}")
