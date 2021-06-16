import numpy as np

from organism import Organism
from structure.linear_structure import LinearStructure


# 直線結構型生物: 採用"直線串接型"的生物，不會因任務不同而個別設立類別
class LinearOrganism(Organism):
    def __init__(self, gene: np.array, n_cell: int = -1, energy: float = 100.0, mutation_rate: float = 0.03):
        super().__init__(gene=gene, n_cell=n_cell, energy=energy, mutation_rate=mutation_rate)

    def call(self):
        pass

    def formStructure(self, gene):
        self.structure = LinearStructure()

        struct_gene = gene[8: 8 + self.n_cell**2].reshape((self.n_cell, self.n_cell))

    def toParams(self):
        pass


if __name__ == "__main__":
    ulo = LinearOrganism(energy=100, mutation_rate=0.03)
    print(f"#gene: {len(ulo.gene)}")
