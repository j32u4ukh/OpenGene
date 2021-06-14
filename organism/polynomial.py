from organism import Organism
from cell import DenseCell


class UnaryLinearOrganism(Organism):
    n_gene = DenseCell.n_gene

    def __init__(self, energy, mutation_rate):
        super().__init__(energy=energy, mutation_rate=mutation_rate)

    def call(self):
        pass


if __name__ == "__main__":
    ulo = UnaryLinearOrganism(energy=100, mutation_rate=0.03)
    print(f"#gene: {len(ulo.gene)}")
