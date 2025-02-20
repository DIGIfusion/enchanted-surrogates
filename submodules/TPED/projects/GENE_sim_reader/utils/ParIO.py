""" ParIO.py: Contains the class to handle reading and writing of parameter files """
import re
import sys
from collections import namedtuple, OrderedDict


class Parameters(object):
    """Parameters class:

    Converts a GENE parameters file to a python dictionary and vice versa
    """

    def __init__(self):
        # dictionary for parameters: {parameter: value}
        self.pardict = OrderedDict()
        # dictionary for recording the namelists the parameters belong to: {parameter: namelist}
        self.nmldict = OrderedDict()
        # keep track of all namelists that have been found
        self.namelists = []
        self.spec_nl = (
            "omn",
            "omt",
            "mass",
            "charge",
            "dens",
            "temp",
            "name",
            "passive",
            "kappa_n",
            "kappa_T",
            "LT_center",
            "Ln_center",
            "LT_width",
            "Ln_width",
            "prof_type",
            "src_prof_type",
            "src_amp",
            "src_width",
            "src_x0",
            "delta_x_n",
            "delta_x_T",
            "prof_file",
        )
        self.boolstr_t = [".T.", ".t.", "T", "t", ".true."]
        self.boolstr_f = [".F.", ".f.", "F", "f", ".false."]

    @staticmethod
    def clearcomments(variable):
        regex = re.compile(r"\s*([-+\'\"\[\];.,/a-zA-Z0-9_\s*]*)\s*!?\s*(.*)")
        result = regex.search(variable)
        if result and result.group(2)[:4] != "scan":
            return regex.search(variable).group(1)
        else:
            return variable

    def Read_Pars(self, path):
        """Read parameters file and make it a dict"""
        # counts species namelists
        countspec = 0
        try:
            with open(path, "r") as parfile:
                # Search file for parameters using regular expressions
                for line in parfile:
                    # Exclude commented lines
                    if re.search(r"\s*!\w*\s*=.*", line) is None:
                        # Check for and count species namelists
                        if re.search(r"^\s*&(.*)", line):
                            # if namelist belongs to a species, append its number to the namelist
                            if re.search(r"^\s*&(.*)", line).group(1) == "species":
                                countspec += 1
                                nml = re.search(r"^\s*&(.*)", line).group(1) + str(
                                    countspec
                                )
                            else:
                                nml = re.search(r"^\s*&(.*)", line).group(1)
                            if nml not in self.namelists:
                                self.namelists.append(nml)
                    # Search lines for <parameter> = <value> patterns
                    p = re.compile(r"^\s*(.*)\s*=\s*(.*)")
                    m = p.search(line)
                    # Pick matching lines and build parameter dictionary
                    if m:
                        if m.group(1).strip() in self.spec_nl:
                            self.pardict[m.group(1).strip() + str(countspec)] = m.group(
                                2
                            )
                            self.nmldict[m.group(1).strip() + str(countspec)] = nml
                        else:
                            self.pardict[m.group(1).strip()] = m.group(2)
                            self.nmldict[m.group(1).strip()] = nml
        except IOError:
            sys.exit("ParIO: ReadPars: could not read parameters file")
        # clear the comments from all variables, cast some strings to integers and floats
        for item in self.pardict:
            self.pardict[item] = self.clearcomments(self.pardict[item])
            try:  # Can it be converted to int?
                self.pardict[item] = int(self.pardict[item])
            except ValueError:
                try:  # No, but can it be converted to float?
                    self.pardict[item] = float(self.pardict[item])
                except ValueError:
                    pass
            if self.pardict[item] in self.boolstr_t:  # cast switches to boolean values
                self.pardict[item] = True
            elif self.pardict[item] in self.boolstr_f:
                self.pardict[item] = False
            # generate class attributes from all valid parameters
            # for k,v in self.pardict.iteritems():
            #    if len(k.split())==1 and k[0]!='!':vars(self)[k]=v
            if isinstance(self.pardict[item], str):
                strip_value = self.pardict[item].strip("'")
                strip_value = strip_value.strip('"')
                self.pardict[item] = strip_value

    def Write_Pars(self, path):
        """Take the dict and write a GENE parameters file"""
        parfileout = open(path, "w")
        specflag = 0
        for item in self.namelists:
            specflag = 0
            if item[0:-1] == "species":
                specflag = 1
                parfileout.write("&" + item[0:-1] + "\n")
            else:
                parfileout.write("&" + item + "\n")
            if specflag:
                [
                    parfileout.write(par[0:-1] + "=" + self.pardict[par] + "\n")
                    for par in self.pardict.keys()
                    if self.nmldict[par] == item
                ]
            else:
                [
                    parfileout.write(par + "=" + self.pardict[par] + "\n")
                    for par in self.pardict.keys()
                    if self.nmldict[par] == item
                ]
            parfileout.write("/\n\n")

    def asnamedtuple(self):
        """Return Parameters as a named tuple for easier usage"""
        # We have to ignore parameters with whitespace (in info nml)
        valid = OrderedDict()
        for item in self.pardict:
            if " " not in item:
                valid[item] = self.pardict[item]
        return namedtuple("ParTuple", valid.keys())(*valid.values())
