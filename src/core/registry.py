"""
Plugin registry for dynamic component discovery and registration.

This module provides a centralized registry for datasets, models, trainers,
and evaluators, enabling runtime discovery and instantiation.
"""

from typing import Any, Callable, Dict, Optional, Type


class Registry:
    """
    A centralized registry for pluggable components.
    
    The Registry allows registration and retrieval of classes/functions
    by name within named categories. This enables configuration-driven
    instantiation of components.
    
    Example:
        >>> registry = Registry()
        >>> @registry.register("dataset", "safety")
        ... class SafetyDataset:
        ...     pass
        >>> dataset_cls = registry.get("dataset", "safety")
        >>> dataset = dataset_cls()
    """
    
    _instance: Optional["Registry"] = None
    
    def __new__(cls) -> "Registry":
        """Singleton pattern for global registry access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registries: Dict[str, Dict[str, Any]] = {}
        return cls._instance
    
    def register(self, category: str, name: str) -> Callable[[Type], Type]:
        """
        Register a class or function under a category and name.
        
        Args:
            category: The category to register under (e.g., "dataset", "model").
            name: The unique name within the category.
        
        Returns:
            A decorator that registers the class/function.
        
        Example:
            >>> @registry.register("dataset", "safety")
            ... class SafetyDataset:
            ...     pass
        """
        def decorator(cls: Type) -> Type:
            if category not in self._registries:
                self._registries[category] = {}
            if name in self._registries[category]:
                raise ValueError(
                    f"'{name}' already registered in category '{category}'"
                )
            self._registries[category][name] = cls
            return cls
        return decorator
    
    def get(self, category: str, name: str) -> Any:
        """
        Retrieve a registered component by category and name.
        
        Args:
            category: The category to look in.
            name: The name of the component.
        
        Returns:
            The registered class or function.
        
        Raises:
            KeyError: If the category or name is not found.
        """
        if category not in self._registries:
            raise KeyError(f"Category '{category}' not found in registry")
        if name not in self._registries[category]:
            available = list(self._registries[category].keys())
            raise KeyError(
                f"'{name}' not found in category '{category}'. "
                f"Available: {available}"
            )
        return self._registries[category][name]
    
    def list_category(self, category: str) -> list[str]:
        """
        List all registered names in a category.
        
        Args:
            category: The category to list.
        
        Returns:
            List of registered names.
        """
        if category not in self._registries:
            return []
        return list(self._registries[category].keys())
    
    def list_categories(self) -> list[str]:
        """
        List all registered categories.
        
        Returns:
            List of category names.
        """
        return list(self._registries.keys())
    
    def clear(self) -> None:
        """Clear all registrations. Useful for testing."""
        self._registries.clear()


# Global registry instance
registry = Registry()


def register(category: str, name: str) -> Callable[[Type], Type]:
    """
    Convenience function to register with the global registry.
    
    Args:
        category: The category to register under.
        name: The unique name within the category.
    
    Returns:
        A decorator that registers the class/function.
    
    Example:
        >>> from src.core.registry import register
        >>> @register("dataset", "safety")
        ... class SafetyDataset:
        ...     pass
    """
    return registry.register(category, name)


def get_component(category: str, name: str) -> Any:
    """
    Convenience function to get a component from the global registry.
    
    Args:
        category: The category to look in.
        name: The name of the component.
    
    Returns:
        The registered class or function.
    """
    return registry.get(category, name)
