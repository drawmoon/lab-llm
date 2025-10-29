## Prefer using cast("Foo", foo) over cast(Foo, foo)

**运行时性能优化 (Runtime Performance)**

这是最直接的技术原因。

* **`cast(Foo, foo)`**：在 Python 运行时，解释器必须先求值第一个参数。如果 `Foo` 是一个普通的类，开销还很小；但如果它是一个复杂的泛型（例如 `dict[str, list[int]]`），Python 必须在运行时构建这个类型对象。在一个频繁调用的“热循环” (hot loop) 中，这种重复的类型构建会带来不必要的性能损耗。
* **`cast("Foo", foo)`**：传递字符串字面量几乎是零开销的（仅涉及 `LOAD_CONST` 指令）。Python 运行时完全忽略这个字符串的内容，只有静态类型检查器（如 MyPy, Pyright）会去解析它。

**避免循环导入与前向引用 (Circular Imports & Forward Refs)**

这是架构层面的原因，也是 Ruff 这一类工具推崇的“最佳实践”。

* 为了避免循环引用，我们经常把类型导入放在 `if TYPE_CHECKING:` 块中。
* 如果在运行时使用 `cast(Foo, ...)`，那么 `Foo` 必须在运行时被导入，这会破坏 `if TYPE_CHECKING:` 的隔离作用，导致运行时再次出现 `ImportError` 或循环依赖。
* 使用 `cast("Foo", ...)` 允许你在运行时完全不需要导入 `Foo` 类，真正实现了运行时代码与类型提示代码的解耦。

**代码风格的一致性 (Consistency)**

这是 Ruff 作为 Linter 的设计哲学。

* 在大型项目中，由于上述第 2 点的原因，你总会遇到一些情况必须使用字符串（例如前向引用）。
* 结果就是代码库中混杂着 `cast(Foo, foo)` 和 `cast("Foo", foo)` 两种写法。
* TC006 规则认为：既然字符串写法在任何场景下都通用（既支持已定义的类，也支持未定义的类），那么强制统一使用字符串写法可以消除视觉上的不一致，并防止未来重构时因为移动代码位置而意外引入 `NameError`。

