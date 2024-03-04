from typing import List, Literal, Union, Optional
from typing import TYPE_CHECKING

import json

from ..Base import Hashable
from .Parameter import Parameter, ParameterListener
from .Variable import Variable, VariableInstance

if TYPE_CHECKING:
    from ..modelskeleton import ModelTemplate
    from .Layer import Layer


class LayerModel(Hashable):
    """
    LayerModel Class

    A class representing a model used in a layer of a neural network.

    Attributes:
        name (str): The name of the layer model.
        template (ModelTemplate): The model template associated with the layer model.
        input_variables (list): List of input variables attached to the layer model.
        output_variables (list): List of output variables attached to the layer model.
        attached_layers (list): List of neural layers attached to the layer model.

    Methods:
        __init__(name: str, model_template: ModelTemplate)
            Initializes a new instance of the LayerModel class.

        attach_variables(variables: Variable | List[Variable], where: Literal['in', 'out'])
            Attaches the variables of the model.

        attach_layers(layers: Layer | List[Layer])
            Attaches the given neural layers to the model.

        detach_layer(layer: Layer)
            Detaches the specified neural layer from the layer model.

        get_input_variables() -> list
            Returns the list of input variables attached to the layer model.

        get_output_variables() -> list
            Returns the list of output variables attached to the layer model.

        get_attached_layers() -> list
            Returns the list of neural layers attached to the layer model.

        __format__(format_spec: str) -> str
            Returns a formatted string representation of the layer model.
    """

    def __init__(
        self,
        name: str,
        input_variables: Optional[List[Variable]] = None,
        output_variables: Optional[List[Variable]] = None,
        parameters: Optional[List[Parameter]] = None
    ):
        """
        Initializes a new instance of the LayerModel class.

        Parameters
        ----------
        name : str
            The name of the layer model.
        template : ModelTemplate
            The model template to use for the model.
        """
        super(LayerModel, self).__init__()
        self.name = name
        self.input_variables = []
        self.output_variables = []
        self.parameters = []
        self.named_variables = {}
        self.named_parameters = {}
        if input_variables:
            self.attach_variables(input_variables, where="in")
        if output_variables:
            self.attach_variables(output_variables, where="out")
        if parameters:
            self.attach_parameters(parameters)
        self.attached_layers = []

    def attach_variables(
        self,
        variables: Union[Variable, List[Variable]],
        where: Optional[Literal["in", "out"]] = "in",
    ):
        """
        Attaches the variables of the model.

        Parameters
        ----------
        variables : Variable | List[Variable]
            The variables to attach.
        where : Literal['in', 'out']
            The place to attach the variables. Either "in" for input variables or "out" for output variables.
            This parameter is defaulted to "in" and is not accounted for in input and output models.

        Returns
        -------

        """
        if isinstance(variables, Variable):
            if where == "in":
                self.input_variables.append(variables)
            elif where == "out":
                self.output_variables.append(variables)
            else:
                raise ValueError(
                    "incorrect place to put the variable : {where}".format(where=where)
                )
            variables.attach_model(self)
            if self.named_variables.get(variables.name, None) is None:
                self.named_variables[variables.name] = variables
            else:
                raise KeyError(f"variable {variables.name} already used  in {self.name}")
        elif isinstance(variables, list):
            if where == "in":
                self.input_variables.extend(variables)
            elif where == "out":
                self.output_variables.extend(variables)
            else:
                raise ValueError(
                    "incorrect place to put the variables : {where}".format(where=where)
                )
            for variable in variables:
                variable.attach_model(self)
                if self.named_variables.get(variable.name, None) is None:
                    self.named_variables[variable.name] = variable
                else:
                    raise KeyError(f"variable {variable.name} already used  in {self.name}")

    def attach_parameters(self, parameters: Union[Parameter, List[Parameter]]):
        if isinstance(parameters, Parameter):
            parameters = [parameters]
        self.parameters.extend(parameters)
        for parameter in parameters:
            parameter.attach_parent(self)
            if self.named_parameters.get(parameter.name, None) is None:
                self.named_variables[parameter.name] = parameter
            else:
                raise KeyError(f"parameter name {parameter.name} already used in {self.name}")


    def attach_layer(self, layer: "Layer"):
        if isinstance(layer, Hashable):
            if layer not in self.attached_layers:
                self.attached_layers.append(layer)
                for variable in self.input_variables:
                    variable.make_new_instance(layer)
                for variable in self.output_variables:
                    variable.make_new_instance(layer)
        else:
            raise TypeError(
                "incorrect type for layers to attach : {type}".format(type=type(layer))
            )

    def detach_layer(self, layer: "Layer"):
        if layer in self.attached_layers:
            self.attached_layers.remove(layer)
            for variable in self.input_variables:
                variable.remove_instance(layer)
            for variable in self.output_variables:
                variable.remove_instance(layer)

    def get_input_variables(self):
        return self.input_variables

    def get_output_variables(self):
        return self.output_variables

    def get_attached_layers(self):
        return self.attached_layers

    def __format__(self, format_spec):
        return f"{self.__class__.__name__}-{self.hash}({self.name})"


class InputModel(LayerModel):
    """
    InputModel Class

    The model to use for the input layer of a neural network.

    Attributes:
        name (str): The name of the input model.
        template (ModelTemplate): The model template associated with the input model.
        input_variables (list): List of input variables attached to the input model.
        output_variables (list): List of output variables attached to the input model.
        attached_layers (list): List of neural layers attached to the input model.

    Methods:
        __init__(name: str, model_template: ModelTemplate)
            Initializes a new instance of the InputModel class.

        attach_variables(variables: Variable | List[Variable], where: Literal['in', 'out'])
            Attaches the variables of the model.

        attach_layers(layers: Layer | List[Layer])
            Attaches the given neural layers to the model.

        detach_layer(layer: Layer)
            Detaches the specified neural layer from the input model.

        get_input_variables() -> list
            Returns the list of input variables attached to the input model.

        get_output_variables() -> list
            Returns the list of output variables attached to the input model.

        get_attached_layers() -> list
            Returns the list of neural layers attached to the input model.

        __format__(format_spec: str) -> str
            Returns a formatted string representation of the input model.
    """

    def __init__(
        self,
        name: str,
        output_variables: Optional[List[Variable]] = None,
    ):
        """
        Initializes a new instance of the InputModel class.

        Parameters
        ----------
        name : str
            The name of the input model.
        output_variables : Optional[List[Variable]]
            The output variables to attach to the model.

        """
        super(InputModel, self).__init__(
            name, input_variables=None, output_variables=output_variables
        )

    def attach_variables(
        self,
        variables: Union[Variable, List[Variable]],
        where: Optional[Literal["in", "out"]] = "in",
    ):
        """
        Attaches the variables of the model. This models neglects the "where" parameter and only attaches the variables
        as output variables.

        Parameters
        ----------
        variables : Variable | List[Variable]
            The variables to attach.
        where : Literal['in', 'out']
            Unused parameter

        Returns
        -------

        """
        if isinstance(variables, Variable):
            variables.make_instantiable(False)

        elif isinstance(variables, list):
            self.output_variables.extend(variables)
            for variable in variables:
                variable.make_instantiable(False)
        LayerModel.attach_variables(self, variables, where="out")

class OutputModel(LayerModel):
    """
    InputModel Class

    The model to use for the input layer of a neural network.

    Attributes:
        name (str): The name of the input model.
        input_variables (list): List of input variables attached to the input model.
        output_variables (list): List of output variables attached to the input model.
        attached_layers (list): List of neural layers attached to the input model.

    Methods:
        __init__(name: str, model_template: ModelTemplate)
            Initializes a new instance of the InputModel class.

        attach_variables(variables: Variable | List[Variable], where: Literal['in', 'out'])
            Attaches the variables of the model.

        attach_layers(layers: Layer | List[Layer])
            Attaches the given neural layers to the model.

        detach_layer(layer: Layer)
            Detaches the specified neural layer from the input model.

        get_input_variables() -> list
            Returns the list of input variables attached to the input model.

        get_output_variables() -> list
            Returns the list of output variables attached to the input model.

        get_attached_layers() -> list
            Returns the list of neural layers attached to the input model.

        __format__(format_spec: str) -> str
            Returns a formatted string representation of the input model.
    """

    def __init__(
        self,
        name: str,
        input_variables: Optional[List[Variable]] = None,
    ):
        """
        Initializes a new instance of the InputModel class.

        Parameters
        ----------
        name : str
            The name of the input model.
        input_variables : Optional[List[Variable]]
            The input variables to attach to the model.
        """
        super(OutputModel, self).__init__(
            name, input_variables=input_variables, output_variables=[]
        )

    def attach_variables(
        self,
        variables: Union[Variable, List[Variable]],
        where: Optional[Literal["in", "out"]] = "in",
    ):
        """
        Attaches the variables of the model. This models neglects the "where" parameter and only attaches the variables
        as input variables.

        Parameters
        ----------
        variables : Variable | List[Variable]
            The variables to attach.
        where : Literal['in', 'out']
            Unused parameter

        Returns
        -------

        """

        if isinstance(variables, Variable):
            variables.make_instantiable(False)

        elif isinstance(variables, list):
            for variable in variables:
                variable.make_instantiable(False)

        LayerModel.attach_variables(self, variables)


class TemplatedModel(LayerModel):
    def __init__(self, name: str, template_source=""):
        super(TemplatedModel, self).__init__(name)
        self.template_source = template_source
        self._props = self.load_template_props(template_source)
        self.template_name = self._props["name"]
        self.python_source = self._props["source"]
        # build variables and parameters

        for variable_name, variable_prop in self._props["variables"].items():
            variable_io = variable_prop['IO']
            variable_dim = variable_prop['dim']
            variable_type = variable_prop.get('type', float)

            variable = Variable(name = variable_name,
                                dimension=variable_dim,
                                data_type=variable_type,
                                variable_io=variable_io)
            self.attach_variables(variable, variable_io)
        for param_name, param_prop in self._props["parameters"].items():
            param_type = param_prop["type"]
            parameter = Parameter(name=param_name,
                                  parameter_type= param_type
                                  )
            self.attach_parameters(parameter)




    def load_template_props(self, source_file: str):
        file = open(source_file, "r")
        text = file.read()
        file.close()

        return json.loads(text)
