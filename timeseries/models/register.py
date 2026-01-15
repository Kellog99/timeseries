from collections.abc import Callable
from typing import TypeVar, Union, Type, Optional

import torch
from pydantic import BaseModel, Field, field_validator
from pydantic.fields import FieldInfo

# This allows to declare the type variable "T"
# It will be used for expressing the type of the class that is registered.
T = TypeVar("T")


def remove_suffix(name: str) -> str:
    """This is a function just to remove the suffixes that are written in the name"""
    for suffix in ["Attack", "Loss", "Metric", "Config"]:
        name = name.replace(suffix, "")
    return name.lower()


class Info(BaseModel):
    class_type: Type[torch.nn.Module] = Field(
        default=...,
        description="The timeseries model to be saved in the register."
    )
    id: str = Field(default=..., description="class type id.")
    name: str = Field(default=..., description="class type name.")
    description: str = Field(
        default="No description passed",
        description="This contains the class' description."
    )

    class Config:
        arbitrary_types_allowed = True


class ModelRegister:
    """
        Generic factory base.
        """
    _register: dict[str, Info] = {}
    _info_type: Type[Info]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._register: dict[str, Info] = {}

    @classmethod
    def __register__(
            cls,
            class_type: type,
            id: str,
            name: str,
            description: str,
            **kwargs) -> None:
        if id not in cls._register:
            cls._register[id] = cls._info_type(
                class_type=class_type,
                id=id,
                name=name,
                description=description,
                **kwargs
            )

    ######################################## decorator ########################################

    @classmethod
    def register(cls,
                 name: str,
                 description: str,
                 id: str = None,
                 **kwargs) -> Callable[[type], type]:

        def decorator(class_type: type) -> type:
            # Override __repr__ to return the name
            class_type.__repr__ = lambda self: name
            if len(kwargs) > 0:
                for key in kwargs:
                    setattr(class_type, key, kwargs[key])
            cls.__register__(
                class_type=class_type,
                name=name,
                description=description,
                id=id if id else remove_suffix(class_type.__name__),
                **kwargs)
            return class_type

        return decorator

    ###########################################################################################

    @classmethod
    def get_information(cls, id: str, exclude: set = {"class_type"}) -> dict:
        if id in cls._register:
            return cls._register[id].model_dump(exclude=exclude)
        else:
            raise ValueError(f"The attack {id} is not registered.")

    @classmethod
    def get_config(
            cls,
            class_id: str,
            **kwargs,
    ) -> BaseModel:
        """Get the configuration instance for the specified attack.

        :param class_id: The attack name.
        :param kwargs: Configuration parameters.
        :return: The attack configuration.
        """
        if class_id in cls._register:
            return cls._register[class_id].class_type.CONFIG_T(**kwargs)
        else:
            raise ValueError(f"The attack {class_id} is not registered.")

    @classmethod
    def get_config_param(
            cls,
            id: str | None,
            attribute_type: Union[str, float, bool, int, list, tuple, None] = None
    ) -> list[tuple[str, FieldInfo]]:
        """List the parameters of a specific attack filtered by their types.

        :param id: The class' id to list parameters for.
        :param attribute_type: The filter for the parameters type to retrieve.

        :return: A dict of parameter names and descriptions.
        """
        ######################### Checking the function's input ################################
        if isinstance(attribute_type, (list, tuple)):
            for element in attribute_type:
                if element not in (str, float, bool, int):
                    raise TypeError(f"The type of {element}, {type(element)}, is not allowed.")
            ########################################################################################

        out = []
        if id:
            for key, value in cls._register[id].class_type.CONFIG_T.model_fields.items():
                if (attribute_type is None) or (value.annotation in attribute_type):
                    out.append((key, value))
        return out

    @classmethod
    def create(
            cls,
            class_id: Optional[str] = None,
            config: Optional[type] = None,
            **kwargs) -> Type[T]:
        """
        Create an instance of the registered class following the configuration file that is passed or the default one (if possible).

        :param class_id: The class type id in the register. if not provided, it will be inferred by the config.
        :param config: The class' configuration file.
        :param kwargs: Parameters to set in the config if not passed, otherwise to update.
        :return: The requested attack instance.
        """

        # this function has the role to instantiate the class that is saved into the register
        # associated to the id
        ################ checking the validity of the query ################
        cnf_id = remove_suffix(type(config).__name__) if config else ""
        if class_id is None:
            if config is None:
                # at least one of these two has to be passed.
                raise ValueError("Either the class_id or the config must be provided")
            else:
                class_id = cnf_id
        else:
            if class_id not in cls._register:
                raise ValueError(f"There are no classes associated to the id {class_id}")
            if config and class_id != cnf_id:
                raise Warning(f"The class' id, {class_id}, differs from the config's id, {cnf_id}")
        ####################################################################

        if config:

            if len(kwargs) > 0:
                # Infer attack type from config class name
                # Update config with kwargs
                for key, value in kwargs.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        else:
            config = cls._register[class_id].class_type.CONFIG_T(**kwargs)

        return cls._register[class_id].class_type(config)

    @classmethod
    def filter(cls, info: Info, **kwargs) -> bool:
        """
        This method has the role to tell whether a class has to be added on the list
        that is returned by the `get_list_classes` or not.
        """
        return True

    @classmethod
    def get_list_classes(
            cls,
            **kwargs
    ):
        return [
            id
            for id, info in cls._register.items()
            if cls.filter(info=info, **kwargs)
        ]
