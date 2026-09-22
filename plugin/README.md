# hermes_laya plugin

Drop-in Hermes plugin for the [Laya](https://github.com/NandhaKishorM/laya)
System 1 decision engine.

```
plugin/
├── __init__.py     # two-step registration via ctx.register_tool() (plugin-owned)
├── plugin.yaml     # plugin metadata (name, version, tags)
└── laya_tool.py    # the tool: schema + handler + check_fn + module-level register()
```

`install.sh` in the repo root copies these three files into
`~/.hermes/plugins/hermes_laya/`. They can also be copied by hand.

Why a plugin and not `tools/`: `hermes update` resets everything inside the
hermes-agent git tree, so locally-added tools wired into `tools/` or
`toolsets.py` vanish on the next update. `~/.hermes/plugins/` lives outside the
tree and survives.

The tool gates on `import laya` via `check_fn`, so a machine without the pip
package simply does not see the tool instead of erroring.
