import numpy as np

from organism import Organism
from structure.linear_structure import LinearStructure


class UnaryLinearOrganism(Organism):
    n_gene = 904268

    def __init__(self, n_gene: int = 904268, energy: float = 100.0, mutation_rate: float = 0.03):
        super().__init__(n_gene=n_gene, energy=energy, mutation_rate=mutation_rate)

    def call(self):
        pass

    @classmethod
    def parseGene(cls, gene: np.array):
        """

        :return:
        """
        n_cell, gene = super().checkGeneNumber(gene)

        if gene is None:
            return None, None
        else:
            structure_gene = gene[:n_cell**2]
            structure_matrix = structure_gene.reshape((n_cell, n_cell))
            structure = LinearStructure()
            structure.parseStructure(matrix=structure_matrix)


if __name__ == "__main__":
    ulo = UnaryLinearOrganism(energy=100, mutation_rate=0.03)
    print(f"#gene: {len(ulo.gene)}")
