from Base.Network import Layer, LayerModel, ModelTemplate, Variable, Network

# load test libraries
import unittest


class TestNeuralNetwork(unittest.TestCase):
    def test_init(self):
        input_variables = [
            Variable.Variable("input", 2, "in", float, instantiable=False)
        ]
        input_model = LayerModel.InputModel("input", input_variables)
        input_layer = Layer.Layer(input_model)

        print(f"generated input layer: {input_layer}")

        output_variables = [Variable.Variable("output", 2, "out", float)]
        output_model = LayerModel.OutputModel("output", output_variables)
        output_layer = Layer.Layer(output_model)

        print(f"generated output layer: {output_layer}")

        input_variables[0].add_linked_variables(output_variables[0])
        output_variables[0].add_linked_variables(input_variables[0])

        network = Network.Network(
            [input_model, output_model],
            [input_layer, output_layer],
            input_layer,
            output_layer,
        )

        print(f"generated network {network}")


if __name__ == "__main__":
    unittest.main()
