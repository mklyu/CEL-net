from typing import Callable, Generic, TypeVar, List, Any

Callback = TypeVar("Callback", bound=Callable)


class Event(Generic[Callback]):
    """
        Event class that attempts to mimic C# events.

        Usage: add callables to instances, call with args to run.

        Example:

        event = Event[Callable[[argTypes],retType]]()
        event += my_function_with_args
        event(args)

        Optional: Subclass this class and redefine __call__ with specific arguments (for arg check)

        Example:

        Class MyEvent(Event[Callable[[argTypes],retType]]):
            def __call__(self, specificArgs...):
                super().__call__(specificArgs ...)
        
        event = MyEvent()
        event += my_function_with_args
        event(specificArgs)

        For a fine-grained control, use the [targets] list inside the Event instances

    """

    def __init__(self):
        self.events: List[Callback] = []

    def __call__(self, *a, **kw):
        for event in tuple(self.events):
            event(*a, **kw)

    def __iadd__(self, event: Callback):
        self.events.append(event)
        return self

    def __isub__(self, event: Callback):
        while event in self.events:
            self.events.remove(event)
        return self

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        def gen():
            for event in self.events:
                yield event

        return gen()

    def __getitem__(self, key):
        return self.events[key]

    def Flush(self):
        self.events = []


if __name__ == "__main__":

    def zeroArgEvent():
        print("Ahoy")

    def oneArgEvent(name: str):
        print("Ahoy " + name + "!")

    class MyEventType(Event[Callable[[str], Any]]):
        def __call__(self, name: str) -> Any:
            super().__call__(name)

    # use case 1:
    callableEvent = Event[Callable[[str], Any]]()

    # callableEvent += zeroArgEvent # IDE should hint error

    callableEvent += oneArgEvent
    # no argument hinting in this case
    callableEvent("some generic event!")

    myEvent = MyEventType()

    # callableEvent += zeroArgEvent # IDE should hint error
    
    myEvent += oneArgEvent
    # IDE should hint arguments
    myEvent("some specific event")
