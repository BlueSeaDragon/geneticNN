from Base.Network import Layer, LayerModel, ModelTemplate, Variable, Network

# load test libraries
import unittest
from pathlib import Path


class TestIONetwork(unittest.TestCase):
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

class TestLinearNetwork(unittest.TestCase):
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

        print(f"generating Linear Layer")

        linear_file = "src/basic_templates/Linear.json"
        print(Path(linear_file).absolute())
        linear_model = LayerModel.TemplatedModel("linear",linear_file)
        linear_layer = Layer.Layer(linear_model)

        input_link = Layer.LayerIOLink(input_layer, input_model.named_variables["input"], linear_layer, "X")
        output_link = Layer.LayerIOLink(linear_layer, "Y", output_layer, "output")

        input_layer.set_io_links(input_link)
        linear_layer.set_io_links(input_link)
        linear_layer.set_io_links(output_link)
        output_layer.set_io_links(output_link)

        network = Network.Network(
            [input_model, linear_model, output_model],
            [input_layer, linear_layer,  output_layer],
            input_layer,
            output_layer,
        )

        print(f"generated network {network}")

class TestMultipleLinearNetwork(unittest.TestCase):
    def test_init(self):
        input_variables = [
            Variable.Variable("input", 2, "out", float, instantiable=False)
        ]
        input_model = LayerModel.InputModel("input", input_variables)
        input_layer = Layer.Layer( input_model,"input_layer",)

        print(f"generated input layer: {input_layer}")

        output_variables = [Variable.Variable("output", 2, "in", float)]
        output_model = LayerModel.OutputModel("output", output_variables)
        output_layer = Layer.Layer( output_model,"output_layer",)

        print(f"generated output layer: {output_layer}")

        input_variables[0].add_linked_variables(output_variables[0])
        output_variables[0].add_linked_variables(input_variables[0])

        print(f"generating Linear Layer")

        linear_file = "src/basic_templates/Linear.json"
        print(Path(linear_file).absolute())
        linear1_model = LayerModel.TemplatedModel("linear1",linear_file)
        linear_layer1 = Layer.Layer(linear1_model,"linear1_layer", )
        linear_layer2 = Layer.Layer(linear1_model,"linear1_layer2",)
        linear2_model = LayerModel.TemplatedModel("linear2", linear_file)
        linear2_layer = Layer.Layer( linear2_model,"linear2_layer")

        add_file = "src/basic_templates/Add.json"
        Add_model = LayerModel.TemplatedModel("Add", add_file)
        add_layer = Layer.Layer( Add_model,"add_layer")

        input_link = Layer.LayerIOLink(input_layer, "input", linear_layer1, "X")
        input_link2 = Layer.LayerIOLink(input_layer, "input", linear2_layer, "X")
        add_link1 = Layer.LayerIOLink(linear2_layer, "Y", add_layer, "X1")
        middle_link1 = Layer.LayerIOLink(linear_layer1, "Y", linear_layer2, "X")
        add_link2 = Layer.LayerIOLink(linear_layer2, 'Y', add_layer, "X2")
        output_link= Layer.LayerIOLink(add_layer, "Y", output_layer, "output")

        input_link.make_link()
        input_link2.make_link()
        add_link1.make_link()
        middle_link1.make_link()
        add_link2.make_link()
        output_link.make_link()

        network = Network.Network(
            [input_model, linear1_model, linear2_model, output_model, Add_model],
            [input_layer, linear_layer1, linear_layer2, linear2_layer,add_layer,  output_layer],
            input_layer,
            output_layer,
        )

        print(f"generated network {network}")
        print(f"network height: ")
        height = network.layers_heights
        for key, height in height.items():
            name = key.name
            print(f"{name}: {height}")


if __name__ == "__main__":
    unittest.main()
