from pint import UnitRegistry


ureg = UnitRegistry()
ureg.define("kev = 1000 * electronvolt")
ureg.define("kpa = 1000 * pascal")
ureg.define("kv = 1000 * volts")

# Q_ = ureg.Quantity