from collections.abc import Callable
from typing import Any, ClassVar, Generic, Literal, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field
from pydantic.fields import FieldInfo

T = TypeVar("T")

Domain = Literal[
    "forecasting", "technical_analysis", "volatility", "filtering",
    "dependence", "statistics", "risk", "returns",
]


def remove_suffix(
        name: str
) -> str:
    """Derive a default id from a class or configuration name."""
    return name.lower().removesuffix("model")


class Configuration(BaseModel):
    model_config = ConfigDict(extra = "forbid", validate_default = True, allow_inf_nan = False)


class Info(BaseModel, Generic[T]):
    model_config = ConfigDict(arbitrary_types_allowed = True, frozen = True)

    class_type: type[T] = Field(description = "Implementation class saved in the registry.")
    id: str = Field(min_length = 1, description = "Unique identifier in this registry.")
    name: str = Field(min_length = 1, description = "Human-readable component name.")
    description: str = Field(min_length = 1, description = "Purpose and behavior of the component.")
    domain: Domain = Field(description = "Application domain of the component.")

    @property
    def config_type(
            self
    ) -> type[BaseModel]:
        return cast(type[BaseModel], getattr(self.class_type, "CONFIG_T"))


class Factory(Generic[T]):
    _register: ClassVar[dict[str, Info]] = {}
    _info_type: ClassVar[type[Info]] = Info

    def __init_subclass__(
            cls,
            **kwargs: Any
    ) -> None:
        super().__init_subclass__(**kwargs)
        cls._register = {}

    @classmethod
    def _load_builtins(
            cls,
            class_id: str | None = None
    ) -> None:
        """Concrete factories may import their built-in implementations on demand."""

    @classmethod
    def __register__(
            cls,
            class_type: type[T],
            id: str,
            name: str,
            description: str,
            domain: Domain,
            **metadata: Any,
    ) -> None:
        key = id.lower()
        if key in cls._register:
            raise ValueError(f"Duplicate registration: {key}")
        config_type = getattr(class_type, "CONFIG_T", None)
        if not isinstance(config_type, type) or not issubclass(config_type, BaseModel):
            raise TypeError(f"{class_type.__name__} must declare a Pydantic CONFIG_T.")
        cls._register[key] = cls._info_type(
            class_type = class_type, id = key, name = name, description = description,
            domain = domain, **metadata,
        )

    @classmethod
    def register[ClassT: type](
            cls,
            name: str,
            description: str,
            id: str | None = None,
            *,
            domain: Domain,
            **metadata: Any,
    ) -> Callable[[ClassT], ClassT]:
        """Register a class, defaulting its id to its lowercase name without a trailing 'model'."""

        def decorator(
                class_type: ClassT
        ) -> ClassT:
            cls.__register__(
                class_type = cast(type[T], class_type), id = remove_suffix(class_type.__name__) if id is None else id,
                name = name, description = description, domain = domain, **metadata,
            )
            return class_type

        return decorator

    @classmethod
    def get_info(
            cls,
            id: str
    ) -> Info[T]:
        key = id.lower()
        cls._load_builtins(key)
        if key not in cls._register:
            raise ValueError(f"Unknown component {id!r}.")
        return cls._register[key]

    @classmethod
    def get_information(
            cls,
            id: str,
            exclude: set[str] | None = None
    ) -> dict[str, Any]:
        return cls.get_info(id).model_dump(exclude = {"class_type"} if exclude is None else exclude)

    @classmethod
    def get_config(
            cls,
            class_id: str,
            **kwargs: Any
    ) -> BaseModel:
        return cls.get_info(class_id).config_type(**kwargs)

    @classmethod
    def get_config_param(
            cls,
            id: str,
            attribute_type: type | list[type] | tuple[type, ...] | None = None,
    ) -> list[tuple[str, FieldInfo]]:
        types = (attribute_type,) if isinstance(attribute_type, type) else attribute_type
        if types is not None and not all(isinstance(item, type) for item in types):
            raise TypeError("attribute_type must contain Python types.")
        return [(key, field) for key, field in cls.get_info(id).config_type.model_fields.items()
                if types is None or field.annotation in types]

    @classmethod
    def filter(
            cls,
            info: Info[T],
            domain: Domain | set[Domain] | None = None,
            **kwargs: Any
    ) -> bool:
        return domain is None or info.domain in ({domain} if isinstance(domain, str) else domain)

    @classmethod
    def get_list_classes(
            cls,
            domain: Domain | set[Domain] | None = None,
            **kwargs: Any
    ) -> list[str]:
        cls._load_builtins()
        return [key for key, info in cls._register.items() if cls.filter(info, domain = domain, **kwargs)]

    @classmethod
    def create(
            cls,
            class_id: str | None = None,
            config: BaseModel | None = None,
            **kwargs: Any,
    ) -> T:
        """Validate constructor arguments and create a registered implementation.

        Explicit overrides are revalidated without mutating the supplied config.
        Constructor keyword arguments match the fields of CONFIG_T.
        """
        if class_id is None:
            if config is None:
                raise ValueError("Either class_id or config must be provided.")
            cls._load_builtins()
            matches = [info.id for info in cls._register.values()
                       if type(config) is info.config_type]
            if not matches:
                matches = [info.id for info in cls._register.values()
                           if isinstance(config, info.config_type)]
            if len(matches) != 1:
                raise ValueError("Provide a class_id when the configuration does not identify exactly one component.")
            class_id = matches[0]
        info = cls.get_info(class_id)
        config_type = info.config_type
        if config is not None and not isinstance(config, config_type):
            raise TypeError(f"{class_id} requires {config_type.__name__}.")
        values = config.model_dump() if config is not None else {}
        validated = config_type.model_validate(values | kwargs)
        return info.class_type(**validated.model_dump())
