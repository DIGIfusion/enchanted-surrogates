"""
# runners/GeneScanRunner.py

Defines the GeneScanRunner class for running the GENE code by using the GENE scan functionality.

"""

# import numpy as np
from .base import Runner
from parsers.GENEparser import GENEparser
import subprocess
import sys, os
import warnings
from dask.distributed import print
from time import sleep
import time


class GeneScanRunner(Runner):
   """
   Class for running the GENE code using the GENE scan functionality.

   Methods:
      __init__(*args, **kwargs)
         Initializes the GENErunner object.
      single_code_run(params: dict, run_dir: str) -> dict
         Runs a single GENE code simulation.

   """
   def __init__(self, executable_path:str, scanscript_path:str, return_mode='growthrate', time:str='12:00:00', account='project_462000451', ssh_command_to_login_node='ssh -i ~/.ssh/lumi-key uan03', gene_dir=None, n_jobs=1, base_parameters_file_path=None, *args, **kwargs):
      """
      Initializes the GENErunner object.

      Args:
         *args: Variable length argument list.
         **kwargs: Arbitrary keyword arguments.
         return_mode (str): Either 'deeplasma' or 'growthrate'. This changes what will be returned.
      """
      self.base_parameters_file_path = base_parameters_file_path
      self.executable_path = executable_path
      self.base_run_dir = kwargs['base_run_dir']
      self.scanscript_path=scanscript_path
      self.base_parameters_file_path = kwargs.get('base_parameters_file_path', None)
      self.parser = GENEparser()
      self.return_mode = return_mode
      self.time = time
      self.n_jobs = n_jobs
      self.account = account
      self.ssh_command_to_login_node = ssh_command_to_login_node
      self.gene_dir = gene_dir
      self.scale_to_num_params = kwargs.get('scale_to_num_params', False) 
      # self.sbatch_string = self.get_sbatch_string()  
   def get_sbatch_string(self):
      return f'''#!/bin/bash -l
## LUMI-C (CPU partition) submit script template
## Submit via: sbatch submit.cmd (parameters below can be overwritten by command line options)
#SBATCH -t {self.time}                # wallclock limit
#SBATCH -N {self.n_jobs}                       # total number of nodes, 2 CPUs with 64 rank each
#SBATCH --ntasks-per-node=128      # 64 per CPU (i.e. 128 per node). Additional 2 hyperthreads disabled
#SBATCH --mem=200GB                    # Allocate all the memory on each node
#SBATCH -p standard                # all options see: scontrol show partition
#SBATCH -J GENE                    # Job name
#SBATCH -o ./GENE.out ##%x.%j.out
#SBATCH -e ./GENE.err ##%x.%j.err
# #uncomment to set specific account to charge or define SBATCH_ACCOUNT/SALLOC_ACCOUNT globally in ~/.bashrc
#SBATCH -A {self.account}

export MEMORY_PER_CORE=1800

## set openmp threads
export OMP_NUM_THREADS=1

#do not use file locking for hdf5
export HDF5_USE_FILE_LOCKING=FALSE

set -x
# run GENE
# srun -l -K -n $SLURM_NTASKS ./gene_lumi_csc

# run scanscript
./scanscript --np $SLURM_NTASKS --ppn $SLURM_NTASKS_PER_NODE --mps 4 --syscall='srun -l -K -n $SLURM_NTASKS {self.executable_path}'

set +x
'''
   def pre_run(self, base_run_dir, params:list, *args, **kwargs):
      print('PERFORMING PRE RUN')
      # Since this is a scan runner the pre_run is actually where the code is ran
      # single_code_run is then where the monitoring and parsing of output is triggered
      if self.scale_to_num_params:
         self.n_jobs = len(params)
      self.parser.write_scan_file(run_dir=base_run_dir, params=params,
                                 base_parameters_file_path=self.base_parameters_file_path, 
                                 n_jobs = self.n_jobs)
      with open(os.path.join(base_run_dir, 'submit.cmd'), 'w') as file:
         file.write(self.get_sbatch_string())
      # self.parser.write_sbatch(run_dir, sbatch_string, wallseconds)
      scanscript_path = os.path.join(base_run_dir, 'scanscript')
      print('WRITING SCANSCRIPT')
      with open(scanscript_path,'w') as file:
         file.write(self.get_scan_script())
      os.system(f'chown --reference={self.executable_path} {scanscript_path}')
      os.system(f'chmod 777 {scanscript_path}')
      print('SUBMITTING SBATCH FOR SCAN')
      print('FOR STD_OUT SEE:', os.path.join(base_run_dir,'GENE.out'))
      #   run_command = f"salloc --mem 0 -A {self.account} --partition standard --ntasks 128 --nodes {self.n_jobs} --time {self.time} --out {os.path.join(run_dir,'GENE.out')} && srun submit.cmd"
      #   os.system(run_command)
      # os.chdir(base_run_dir)
      print('BASE RUN DIR:',self.base_run_dir)
      os.system(f"{self.ssh_command_to_login_node} 'cd {self.base_run_dir} && sbatch submit.cmd'")
      
   def single_code_run(self, run_dir:str, params:dict=None, index=None, *args, **kwargs):
      """
      Usually this function runs a single simulation. In this scan runner the simulation
      is ran in the pre_run function and this function monitors a single scan element
      to see when it is finnished to then parse the output and return the result

      Args:
         params (dict): Dictionary containing parameters for the code run.
         run_dir (str): Directory path where the run command must be called from.
         index int: This informs the function which scan element to monitor, it is
                     converted into a gene suffix.

      Returns:
         (str): Containing comma seperated values of interest parsed from the GENE output 
      """
      suffix = '_'+str(index+1).zfill(4)        
      if index==None:
         raise ValueError('''This is the scan runner, it needs an index to get the gene suffix to track.
                           Bear in mind the index starts at 0 and the suffixes start at 0001''')       
      ready = False
      print('WAITING FOR scanfiles0000 to be made and gene_status file to be made, suffix:',suffix)
      while not ready:
         while 'scanfiles0000' not in os.listdir(self.base_run_dir):
            sleep(5)
         latest_scan_dir = self.parser.latest_scanfiles_dir(self.base_run_dir)
         while not os.path.exists(os.path.join(latest_scan_dir, 'in_par','gene_status')):
            sleep(5)
         ready=True
      
      latest_scan_dir = self.parser.latest_scanfiles_dir(self.base_run_dir)

      finished = False
      i = 0
      while not finished:
         sec=5
         print('WAITING FOR SUFFIX',suffix,'TO FINNISH:',sec*i,'sec')
         status = self.parser.check_status(scanfiles_dir=latest_scan_dir)
         if status[index] == 'f':
            finished=True
         sleep(5)
         i+=1

      # read relevant output values
      if self.return_mode=='deeplasma':
         # required_files = self.parser.read_output(latest_scan_dir, suffix, get_required_files=True)
         # for file in required_files:
         #    while not os.path.exists(os.path.join(latest_scan_dir,file+suffix)):
         #       sleep(1)
         
         #KILL KILL there is something in this read_output that is killing the workers
         output = self.parser.read_output(latest_scan_dir,suffix)
         output = [str(v) for v in output]
      elif self.return_mode=='growthrate':
         # while not os.path.exists(os.path.join(latest_scan_dir,'omega'+suffix)):
         #       sleep(1)
         ky, growthrate, frequency = self.parser.read_omega(latest_scan_dir, suffix)
         output = [str(growthrate)]
            
      params_list = [str(v) for k,v in params.items()]
      return_list = params_list + output
      print('SUFFIX', suffix, 'SUCCESSFULLY PARSED, RESULT:',return_list)
      return ','.join(return_list)
   
   def get_scan_script(run_dir):
      return r'''#!/usr/bin/perl
# for usage type ./scanscript --help and/or view the comments at the end of the file

use strict;
#use warnings;
# add folder to perl path variable @INC
# use lib "../../tools/perl";
use lib "/scratch/project_462000451/gene_enchanted/tools/perl";
use Class::Struct;
use File::Copy;
use Getopt::Long;
use Cwd;
use ReadBackwards ;

# module containing parameter modification routines
use parmod_routines;
use perl_tools;

# define the scan data structure
struct Intervall => {
    name => '@',
    spec => '@',
    start   => '@',
    step => '@',
    end => '@',
    scantype => '@',
    scanindex => '@',
};
our $StrInt = Intervall -> new();

#global variables
our $SCANDIR=" ";
our $OUTDIR=" ";
our $PROBDIR = cwd;
#   temporary parameter file, scanscript is operating on
#   philosophy: in the original parameter file we only modify
#               &scan namelist and result of performance optimization
our $tmpparfile = "$PROBDIR/tmp_parameters";
our $PARINDIR=" ";
our $mysyscall = "";
our $runnum=0;
our $make = (system("gmake -v > /dev/null") > 0) ? "make" : "gmake";
our @datarr;
our @scancoords; #used for scangene
our @autopar_cond ;
#options
my $n_pes=0;              #total number of processes requested for the job
my $force = 0;            #execute all runs, even if error occurs
my $help=0;               #show help
my $long_help=0;          #show advanced help
my $continue_scan=0;      #continue scan in given diagdir
my $test=0;               #do not execute gene (default 0=execute gene)
my $stop=0;               #exit scanscript before gene execution (for testing gene)
my $noeff=0;              #suppress efficiency_scan (for testing gene/ create parameter sets)
my $efficiency=0;         #perform efficiency_scan for different n_procs_sim, then exit
my $min_procs=1;          #minimal number of processes for a efficiency test (default:1)
my $max_n_par_sims=-1;     #maximal number of parallel sims for efficiency test (default: nscans)
my $procs_per_node=0;     #procs per node ;o)
my $ap_switch=1;          #do(1) or do not(0) use autoparallelization result of first run
my $nl_box=0;             #requires ky0_ind scan! set nx0 according to NL box (1): on (0): off
                          # parameters !nl_nx0 = nonlinear number of x modes mus be set, !nl_nx0_max can be set
my $mk_scanlog=0;         #equivalent to --test --continue_scan
my $max_scans=9999;       # maximal recursion depth for routine Traver==maximal number of scans
my $multilist=0;          #multiple lists, scan over list-index 1,2,3... instead of all combos
my $verbose=0;            #some extra prints...
my $reuse_geom=0;         #use geomfile_0001 after first <n_parallel_sims> runs

GetOptions("n_pes:i" => \$n_pes,
      "np:i" => \$n_pes,
      "outdir:s" => \$OUTDIR,
      "o:s" => \$OUTDIR,
      "syscall:s" => \$mysyscall,
      "test!" => \$test,
      "stop!" => \$stop,
      "noeff!" => \$noeff,
      "mk_scanlog!" => \$mk_scanlog,
      "mks!" => \$mk_scanlog,
      "efficiency!" => \$efficiency,
      "eff!" => \$efficiency,
      "min_procs:i" => \$min_procs,
      "max_n_par_sims:i" => \$max_n_par_sims,
      "mps:i" => \$max_n_par_sims,
      "mp:i" => \$min_procs,
      "procs_per_node:i" => \$procs_per_node,
      "ppn:i" => \$procs_per_node,
      "force!" => \$force,
      "f" => \$force,
      "ap_switch:i"=> \$ap_switch,
      "nl_box!"=> \$nl_box,
      "multilist!"=> \$multilist,
      "continue_scan!"=> \$continue_scan,
      "cs!"=> \$continue_scan,
      "verbose!"=> \$verbose,
      "reuse_geom!"=> \$reuse_geom,
      "long_help!"=> \$long_help,
      "help!" => \$help);

if($help==1){
   show_help();
   exit(0);
}

if($long_help==1){
   #show_help();
   show_long_help();
   exit(0);
}
### global variables for efficiency option ###############
our @primes = (2,3,5,7,11,13,17,19,23);
our $max_num_primes = scalar(@primes);
our @n_procs_list ;
our @all_exps ; # prime exponents of all numbers summed up
our @exps;       # help prime exponents
##########################################################

########################################################################
### MAIN PROGRAM: either efficiency scan                               #
###          or  parameter scan:  a)efficiency scan b)write parameters #
###                               c)execute GENE     d)create scan.log #
########################################################################

printf "\n### scanscript with GENE on %.4d processors ###\n\n",$n_pes;

check_options();

prepare_scandir();

our $n_ev=1;
our $comp_type = ReadValue("$tmpparfile","comp_type",1,0);
if ("$comp_type" eq  "0") {$comp_type="'IV'";}
if ("$comp_type" eq  "'EV'"){$n_ev = max(1,ReadValue("$tmpparfile","n_ev",1,0));}


if ($efficiency==1){
   delete_scan_namelist("$tmpparfile");
   efficiency_scan(0);
   copy("$SCANDIR/parameters","$PROBDIR/parameters");
   #if efficiency option is given, only do efficiency scan and exit, output: efficiency.log
   exit(0);
}

if ($continue_scan==0){

   #### a) efficiency test (if possible)
   mkdir("$PARINDIR");
   read_parameters("$tmpparfile","silent");
   if (($noeff==1)or(get_noeff("$PROBDIR/parameters")==1)){
      printf "### skip efficiency test\n";
   }else{
      printf "\n### parallel efficiency and performance optimization\n";
      my $nscans = count_scans("$tmpparfile","verbose");
      efficiency_scan($nscans);
   }

   #### b) write the scan parameter set
   print "\n### writing parameters to in_par/ directory\n";
   create_scanlog();
   $runnum=0;     # initialize new Traver parsing
   Traver(0,$max_scans,"writepar","verbose");      #counts scandims and writes parameters
   set_scan_dims_everywhere();           #scan_dims must be written to parindir and probdir files
   if ($nl_box ==1){set_nl_box();}

}else{ #continue scan

   #  scan namelist and performance settings are not modified and must be correct
   set_chpt_in();
   @scancoords = split(/ /,read_entry("$tmpparfile","scan_dims"));

}

### c call GENE with the prepared parameter set
if ($test==0){
   test_n_parallel_sims();
   print "\n";
   print "\n### preparations done: starting simulation\n";
   execute_gene($n_pes);
}

### d create scan.log (all runs have finished)
$runnum=0;     # initialize new Traver parsing
print "\n### creating scan.log\n";
read_parameters("$tmpparfile","silent");   #needed here to initialize scan dimensions in some cases (e.g. --mks)
create_scanlog();
Traver(0,$max_scans,"scanlog","silent");

finalize_scan();

####################################################################################
##################################Subroutines#######################################
####################################################################################

sub count_scans {
  #parses the StrInt struct to find out the total number of scans
  my $parfile = shift;
  my $silent = shift;
  my $nscans = 1;
  $runnum=0;     # initialize new Traver parsing
  if ($silent ne "silent") {
     printf "   count number of scans...";
  }
  Traver(0,$max_scans,"countscan","silent");
  foreach (@scancoords) {
     $nscans=$nscans*$_;
  }
  if ($silent ne "silent") {
     printf "..done (nscans= $nscans)\n";
  }
  return $nscans;
}

sub check_options {
  if ($continue_scan == 1) {
     print "   continuing scan in current SCANDIR directory:\n";
     print "   valid checkpoints are read\n" ;
  }
  if ($mk_scanlog == 1) {
     # this skips everything except writing the scan.log file
     $test = 1;
     $continue_scan = 1;
  }
  if (($test != 1) and ($mk_scanlog !=1)) {
     #neither test option nor mk_scanlog option: n_pes is required!
     if ($n_pes <= 0) {
        print "error: n_pes = $n_pes <= 0!\n use the scanscript option --n_pes 'number>0'\n";
        exit(0);
     }
  }
}


sub set_nl_box {

  printf "\n\nnx0 values for nonlinear box: \n";
  printf "\$ky0_ind \$ky \$nx0\n";

  opendir(PATH,"$PARINDIR");
  my @content=readdir(PATH);
  closedir(PATH);

  my $file = "";
  my $nx0=0;

  foreach $file(@content){
     if($file=~/parameters/){
        my $parfile = "$PARINDIR/$file";
        my $nl_nx0 = read_entry("$parfile","!nl_nx0");
        my $nl_nx0_max = read_entry("$parfile","!nl_nx0_max");
        if ("$nl_nx0_max" eq "") {$nl_nx0_max = $nl_nx0;}
        my $nl_nx0_min = read_entry("$parfile","!nl_nx0_min");
        if ("$nl_nx0_min" eq "") {$nl_nx0_min = 0;}
        my $nl_nexc = read_entry("$parfile","!nl_nexc");
        my $nl_kymin = read_entry("$parfile","!nl_kymin");
        my $nl_ky0_ind = ReadValue("$parfile","!nl_ky0_ind",1,0);
        if ("$nl_nx0_max" eq ""){$nl_nx0_max = $nl_nx0;}
        foreach ($nl_nx0,$nl_nexc,$nl_kymin,$nl_ky0_ind){
           if ("$_" eq ""){
              print "wrong nl_box, exit at file\n $parfile\n";
              print "--nl_box : this option is for linear runs scanning over ky\n";
              print "using parameters of the nonlinear box\n";
              print "You must specify the nonlinear box by adding the following lines to the parameters file:\n";
              print "!nl_nx0 =     ##  ($nl_nx0) ! the nx0 of the nonlinear box\n";
              print "!nl_nx0_max = ##  ($nl_nx0_max) ! the maximal nx0 for one linear run\n";
              print "!nl_nx0_min = ##  ($nl_nx0_min) ! the minimal nx0 for one linear run default: 0)\n";
              print "!nl_nexc =    ##  ($nl_nexc) ! the nexc of the nonlinear box\n";
              print "!nl_kymin =   ##  ($nl_kymin) ! the kymin of the nonlinear box\n";
              print "!nl_ky0_ind = ##  ($nl_ky0_ind) !scanlist 1,1,32 ! the ky0_ind of the linear runs, do the scan here!\n";
              print "                  (read) current setting in parfile\n";
              exit(0);
           }
        }
        my $ky=$nl_kymin*$nl_ky0_ind;
        my $max_pos_x_modes = int(($nl_nx0-1)/2);
#       this is the calculation of the number of x modes connecting to the ky mode number i
#       the ith ky mode only connects to the every i*nl_nexc th x mode.
#       we count in positive x-direction, the actual number of modes is twice that number (+1 for kx=0 )
#       as long as there are x modes available, count count one up!
        $nx0 = 1;
        for (my $count=1;$count*$nl_nexc*$nl_ky0_ind<=$max_pos_x_modes;$count++){
           $nx0=$count*2+1;
        }
        $nx0 = min($nl_nx0_max,$nx0);
        $nx0 = max($nl_nx0_min,$nx0);
        printf ("%2d    %.3f  %4d\n", $nl_ky0_ind,$ky,$nx0);
        set_entry($parfile,"&box","nx0","$nx0");
        set_entry($parfile,"&box","nky0","1");
        set_entry($parfile,"&box","ky0_ind","1");
        set_entry($parfile,"&box","kymin","$ky");
        set_entry($parfile,"&box","nexc","1");
        set_entry($parfile,"&box","adapt_lx",".t.");
     }
  }
}

sub set_parindir{
# sets the parameter par_in_dir in input parameter $file
# the &scan namelist is created if necessary
  my $file = shift;
  create_scan_namelist("$file");
  set_entry("$file","&scan","par_in_dir","'$PARINDIR'");
}

sub set_scan_dims_everywhere{
  ## scan namelist must be created before (in set_parindir routine)
  my $scandims_string = sprintf "@scancoords";
  ##in probdir
  set_entry("$PROBDIR/parameters","&scan","scan_dims","$scandims_string");
  set_entry("$tmpparfile","&scan","scan_dims","$scandims_string");
  ##also in $PARINDIR
  set_entry_all_files_in_dir("$PARINDIR","&scan","scan_dims","$scandims_string","silent");
  ##in scandir
  set_entry("$SCANDIR/parameters","&scan","scan_dims","$scandims_string");
}

sub set_chpt_in {
  #sets chpt_in to continue scan
  #search for s_checkpoints and checkpoints
  #select newest, prefer s_checkpoints
  opendir(PATH,"$PARINDIR");
  my @content=readdir(PATH);
  closedir(PATH);
  my $file = "";
  my $line = "";
  my $rnum = 0;
  my $parfile = "";
  my $chpt = "";
  my $s_chpt = "";
  my $chpt_in = "";
  my $valid_s_chpt;
  my $valid_chpt;
  if ( $mk_scanlog==0 ) {
     foreach $file(@content){
        if($file=~/parameters_(\d+)$/){
           $rnum = $1;
           $parfile = "$PARINDIR"."/$file";
           $s_chpt = sprintf("%s/s_checkpoint_%.4d",$SCANDIR,$rnum);
           $chpt   = sprintf("%s/checkpoint_%.4d",$SCANDIR,$rnum);
           $valid_s_chpt = (-e $s_chpt and (-s $s_chpt > 6));
           $valid_chpt = (-e $chpt and (-s $chpt > 6));
           #print("size of $s_chpt  ",-s $s_chpt, "  ",$valid_s_chpt,"\n");
           #print("size of $chpt  ",-s $chpt, "  ",$valid_chpt,"\n");
           if ( $valid_s_chpt) {
              if ($valid_chpt){
                 if ($verbose==1) {print("select most recent valid (s_chpt or chpt)\n");}
                 $chpt_in = newest($s_chpt,$chpt); #in doubt use first (s_chpt)
              }else{
                 if ($verbose==1) {print("only s_checkpoint\n");}
                 $chpt_in = $s_chpt;
              }
              set_entry($parfile,"&scan","chpt_in","'$chpt_in'");
              set_entry($parfile,"&in_out","read_checkpoint",".t.");
           }elsif ($valid_chpt){
              if ($verbose==1){print("only checkpoint\n");}
              $chpt_in = $chpt;
              set_entry($parfile,"&scan","chpt_in","'$chpt_in'");
              set_entry($parfile,"&in_out","read_checkpoint",".t.");
           }
           if ($verbose==1){print("+++read  $chpt_in \n");}
        }
     }
  }
}

sub test_n_parallel_sims {
  my $nps = read_entry("$tmpparfile","n_parallel_sims");
  my $n_probs = 1;
  foreach (@scancoords) {
     $n_probs*=$_;
  }
  if ($nps <=0){
     print "n_parallel_sims = $nps <=0, does not make sense!";
     exit(0);
  }elsif ($nps > $n_probs) {
     print "\n\n################################################################\n";
     print "###########  !!!   WARNING   !!!   #############################\n";
     print "################################################################\n";
     print "#   more n_parallel_sims=$nps than scan problems ($n_probs)!      ####\n";
     print "########   !!!   THIS MIGHT BE VERY INEFFICIENT   !!!   ########\n";
     print "################################################################\n\n\n";
  }
}

sub finalize_scan{
  if ($efficiency==1){unlink("$SCANDIR/scan.log")}
  #unlink($tmpparfile);
  my $finished = 1;
  my $line = " ";

  #has gene finished all scans?
  open(FH,"<$PARINDIR/gene_status");
  while ($line=<FH>){;
     if ($line =~ /s+/) {
        $finished = 0;
     }
  }
  close(FH);
  if ($finished == 0){
     print "\n################################################################\n\n";
     print "   gene did not finish all runs, ready for continuation...\n";
     print "   (./scanscript --continue_scan --n_pes <int>)\n\n";
#    original parameter file might not be suited for use of --continue_scan
  }else{
     copy("$SCANDIR/parameters","$PROBDIR/parameters");
     print "\n### scanscript end\n\n";
  }
}

sub initialize_StrInt{
##initialize scan struct with the situation: no scan
  $StrInt = Intervall -> new();

  $StrInt->name(0,"no_scan");
  $StrInt->scantype(0,"none");
  $StrInt->spec(0,0);
  $StrInt->start(0,0);
  $StrInt->step(0,0);
  $StrInt->end(0,0);
  $StrInt->scanindex(0,0);
}

sub print_StrInt{
  my $i = 0;
  while (defined $StrInt->name($i) ){;
     my $name = $StrInt->name($i);
     my $start = $StrInt->start($i);
     my $step=$StrInt->step($i);
     my $end=$StrInt->end($i);
     my $spec=$StrInt->spec($i);
     my $type=$StrInt->scantype($i);
     my $index=$StrInt->scanindex($i);
     print "StrInt $i:\n";
     print "  name $name\n";
     print "  start $start\n";
     print "  step $step\n";
     print "  end $end\n";
     print "  spec $spec\n";
     print "  scantype $type\n";
     print "  scanindex $index\n";
     $i++
  }
  print "number of scan parameters: $i\n";
  #exit(0);
}

sub return_index_StrInt{
  my $searchname=shift;
  my $searchspecies=shift;

  my $i = 0;
  my $foundindex = -1;
  my $name = '';
  my $spec = 0;
  while ((defined $StrInt->name($i)) and ($foundindex lt 0)){
     $name=$StrInt->name($i);
     $spec=$StrInt->spec($i);
     if (($name eq $searchname) and ($spec eq $searchspecies)) {
	 $foundindex=$i;
     }
     $i++
  }

  return $foundindex;
}


####### ROUTINES FOR THE --efficiency OPTION #########################

sub efficiency_scan{
  my $nscans=shift;
  my $n_procs_sim = 0;
  my $n_parallel_sims = 0;
  my $np_gene = 0;
  my $valid = 0;
  my $index = 0;
  my $newfilename = "";
  my $geomfile = '';

  read_parameters("$tmpparfile","verbose");

  ## restrictions for n_parallel_sims for efficiency test prior to actual scan
  if ($efficiency==0){
     #rules for max_n_par_sims
     if ($max_n_par_sims<0){
        $max_n_par_sims = $nscans;  #if not set, set to maximum value
     }
     if ($max_n_par_sims>$nscans){
        $max_n_par_sims = $nscans;  #if set too large, reduce
        if ($max_n_par_sims>$n_pes){
           $max_n_par_sims = $n_pes;#if set too large, reduce
        }
        print "   ATTENTION:  set max. n_parallel_sims to $max_n_par_sims\n";
     }
     my $my_min_procs=max($min_procs,floor($n_pes/$nscans));
     if ($my_min_procs>$min_procs){
        $min_procs = $my_min_procs;
        if ($min_procs>$procs_per_node){
           print "   ATTENTION:  set min. n_procs_sim to $min_procs\n";
        }
     }
  }else{
     #no restrictions for efficiency option:
     $nscans = $n_pes;
     $max_n_par_sims = $nscans;
  }
  $runnum = 0;  #global variable
  my $n_procs_sim_list = eff_initialize($nscans);
  # tests  and switches for autoparallelization and performance optimization:
  @autopar_cond = get_autopar_condition("$PROBDIR/parameters",$ap_switch);
  #@n_procs_list is global...:
  if (($autopar_cond[0]*$autopar_cond[1]*$ap_switch <= 0) and ($#n_procs_list==0)){
     printf "   ...skip efficiency test (n_procs_sim is fixed and perf_opt is not needed)\n";
     $n_procs_sim = $n_procs_list[0];
     $n_parallel_sims = $n_pes/$n_procs_sim;
     edit_scan_namelist("$PROBDIR/parameters",$n_parallel_sims,$n_procs_sim);
     edit_scan_namelist("$tmpparfile",$n_parallel_sims,$n_procs_sim);
  }else{
     foreach $n_procs_sim(@n_procs_list) {
        $n_parallel_sims = $n_pes/$n_procs_sim;
        # this is integer <=> no processor will idle
        #if (not isint($n_parallel_sims)){
        #   print "   no integer n_parallel sims = $n_parallel_sims  -- exit!\n";
        #   exit();
        #}
        if ($efficiency==1) {
           $n_parallel_sims = floor($n_parallel_sims);
           #set the number of MPI procs for GENE
           $np_gene = $n_parallel_sims*$n_procs_sim;
        }else{
           $np_gene = $n_pes;
        }
        $newfilename = "";
        mkdir("$PARINDIR");
        print "   efficiency test: n_parallel_sims = $n_parallel_sims,  n_procs_sim = $n_procs_sim\n";
        set_par_for_eff("$tmpparfile");
        edit_scan_namelist("$tmpparfile",$n_parallel_sims,$n_procs_sim); # in the file to be copied
        edit_scan_namelist("$PROBDIR/parameters",$n_parallel_sims,$n_procs_sim); # and in the file that is read by gene
        $runnum+=1;
        #write parameters to in_par directory
        for ($index=1;$index<=$n_parallel_sims;$index++){
           $newfilename=sprintf("$PARINDIR/parameters_%.4d",$index);
           copy("$tmpparfile","$newfilename");
           ###all the same, but $n_parallel_sims times
        }
        #sleep(0.1); #wait for possible file-system delays
        if ($test == 0 ){
           execute_gene($np_gene);
        }
        #write efflog
        my $filenum = 1;
        write_efflog($filenum);
        #save gene output of performance optimization
        if ($reuse_geom){
           #save first geomfile for later use.
           my $maggeom = ReadValue("$PROBDIR/parameters","magn_geometry",1,0);
           $maggeom=~/\'(\S+)\'/;
           $maggeom=($1);
           $geomfile = sprintf("%s_0001",$maggeom);
           copy("$SCANDIR/$geomfile","$SCANDIR/tmp_geom");
           print("copied geomfile $geomfile to tmp_geom\n");
        }
        $newfilename = sprintf("$SCANDIR/parameters_%.4d_eff",$runnum);
        copy("$SCANDIR/parameters_0001","$newfilename");
        $newfilename = sprintf("$SCANDIR/geneerr.log_%.4d_eff",$runnum);
        copy("$SCANDIR/tmpgene.err",$newfilename);
        $newfilename = sprintf("$SCANDIR/autopar_%.4d_eff",$runnum);
        copy("$SCANDIR/autopar_0001","$newfilename");
        # cleaning up
        system("rm -f $PARINDIR/*");
        for ($index=1;$index<=$n_parallel_sims;$index++){
           system(sprintf("rm -f $SCANDIR/*_%.4d",$index));
           system(sprintf("rm -f $SCANDIR/*_%.4d.h5",$index));
        }
        if ($reuse_geom){
           copy("$SCANDIR/tmp_geom","$SCANDIR/$geomfile");
           print("copied geomfile $geomfile from tmp_geom\n");
        }
     }

     # restore original parameter file. Then, possibly, optimize tmpparfile for efficiency
     copy("$SCANDIR/parameters","$tmpparfile");
     if ($autopar_cond[0]*$autopar_cond[1]*$ap_switch > 0){
     # if possible (all switches ==1), get results of autoparallelization and performance optimization
        ##edit original parameter file : scan parallelization, gene autoparallelization nblocks and perf_vec
        get_opt_parameters("$tmpparfile","perf_opt_on");
     }else{
        #only get scan parallelization (n_procs_sim + n_parallel_sims)
        get_opt_parameters("$tmpparfile","perf_opt_off");
     }
     copy("$tmpparfile","$PROBDIR/parameters");
     #also in regular parameter file read by gene
  }
}

sub  get_opt_parameters {
##  finds the parameters with optimal n_procs_sim and n_parallel_sims from efficiency scan
##  and writes them to the parameter file "$file"

##  note that for the efficiency scan, &box namelist scans are ignored
##  (original entry of parameter file is used), which might lead to
##  nonoptimal n_parallel_sims and n_procs_sim in case of large box scans.
##  there is no way to do this better, since the optimum changes during scan.

##  with the option $perf_opt = "perf_opt_on" also results of performance optimization
##        (&parallelization namelist nblocks and perf_vec) are copied.

  my $parfile = shift;        #target file
  my $perf_opt = shift;
  open(EFF, "<$SCANDIR/efficiency.log");
  my @entry = <EFF>;
  close EFF ;
  my $line;
  my $min_cpu = 1e40;
  my $opt_runnum = 0;
  my $opt_nprocs = 0;
  foreach $line(@entry) {
     if (($line =~ m/\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(.+)\s*\|\s*(.+)/ )&&(not $line =~ m/^#/)){
        if ($4<$min_cpu){
           $min_cpu = $4;
           $opt_runnum = $1;
           $opt_nprocs = sprintf "%d" ,$2 ;
        }
     }
  }
  if ($opt_nprocs>0){
     printf "   optimal parallel efficiency: n_procs_sim = %.4d \n",$opt_nprocs;
  }else{
     printf "   could not determine optimal parallel efficiency!\n";
     printf " ->for possible errors check:\n";
     printf " $SCANDIR/geneerr.log\n";
     printf " $SCANDIR/tmpgene.err\n";
     printf " $PROBDIR/parameters\n";
     printf " or try adding --verbose to the scanscript call\n";

     if ($force == 0){
        #restore original parameter file and exit
        system("cp $SCANDIR/parameters $PROBDIR/parameters");
        exit(0);
     }
  }
# open the parameter file that had optimal performance
  my $opt_parfile = sprintf("$SCANDIR/parameters_%.4d_eff",$opt_runnum);
  if (-e "$opt_parfile") {
     if ("$perf_opt" eq "perf_opt_on"){
        print "   get perf_vec, nblocks and parallelization\n";
        #get the parallelization for optimal performance
        my @list = ("s","w","v","x","y","z");
        foreach (@list){
           my $parname="n_procs_".$_;      #go through coordinates..
           my $new_n_procs=read_entry("$opt_parfile","$parname");
           uncomment_entry("$parfile","$parname",$verbose);
           set_entry("$parfile","&parallelization","$parname","$new_n_procs");
        }
        #get the perf_vec
        my $opt_perf_vec=read_entry("$opt_parfile","perf_vec");
        my $opt_nblocks=read_entry("$opt_parfile","nblocks");
        set_entry("$parfile","&general","perf_vec","$opt_perf_vec");
        set_entry("$parfile","&general","nblocks","$opt_nblocks");
     }
     #get the scan namelist
     create_scan_namelist("$parfile");
     set_entry("$parfile","&parallelization","n_procs_sim","$opt_nprocs");
     my $opt_n_parallel_sims = $n_pes/$opt_nprocs;
     set_entry("$parfile","&parallelization","n_parallel_sims","$opt_n_parallel_sims");
  }else{
     print "$opt_parfile \n not found in get_opt_parameters\n";
     if ($force ==0){
        exit(0);
     }
  }
}


sub get_valid_scan_namelist {
  my $file = shift;
  my $n_parallel_sims= ReadValue("$file","n_parallel_sims",1,0);
  my $n_procs_sim= ReadValue("$file","n_procs_sim",1,0);
  if (($n_parallel_sims <= 0) and ($n_procs_sim <= 0)){
     #print "execute efficiency scan\n\n";
     return(0,$n_parallel_sims,$n_procs_sim);
  }elsif (($n_parallel_sims>0) and ($n_procs_sim<=0)){
     $n_procs_sim = $n_pes/$n_parallel_sims;
     if (isint($n_procs_sim)){
        set_entry("$file","&parallelization","n_procs_sim","$n_procs_sim");
        #valid -->proceed to return(1)
     }else{
        print "error: n_parallel_sims = $n_parallel_sims does not fit with n_pes = $n_pes\n";
        exit(0);
     }
  }elsif (($n_parallel_sims<=0) and ($n_procs_sim>0)){
     $n_parallel_sims = $n_pes/$n_procs_sim;
     if (isint($n_parallel_sims)){
        set_entry("$file","&scan","n_parallel_sims","$n_parallel_sims");
        #valid -->proceed to return(1)
     }else{
        print "error: n_procs_sim = $n_procs_sim does not fit with n_pes = $n_pes\n";
        exit(0);
     }
  }else{
      #valid -->proceed to return(1)
  }
  printf "\n   n_procs_sim=%.4d,  n_parallel_sims=%.4d\n",$n_procs_sim,$n_parallel_sims;
  print "   taken from parameters file.\n\n";
  return(1,$n_parallel_sims,$n_procs_sim);
}

sub set_par_for_eff{
  my $parfile = shift;
  my $collision_op = read_entry($parfile,"collision_op");
  if ("$collision_op" eq "'none'"){
     $collision_op="";
  }
  set_entry("$parfile","&general","comp_type","$comp_type");
  if ("$comp_type" eq "'IV'") {
     set_entry("$parfile","&general","calc_dt",".false.");
     set_entry("$parfile","&general","ntimesteps","5");
     set_entry("$parfile","&general","dt_max","1e-6");
     if ("$collision_op" ne ""){
        set_entry("$parfile","&general","dt_vlasov","1e-6");
        #setting ev_coll, so that RKC1 is chosen if RKCa is selected for splitting
        set_entry("$parfile","&general","ev_coll","1");
     }
     comment_entry("$parfile","which_ev");
     comment_entry("$parfile","n_ev");
     comment_entry("$parfile","ev_max_it");
  }elsif("$comp_type" eq "'EV'"){
     set_entry("$parfile","&general","calc_dt",".false.");
     set_entry("$parfile","&general","ntimesteps","0");
     set_entry("$parfile","&general","dt_max","1e-6"),
     set_entry("$parfile","&general","n_ev","1");
     set_entry("$parfile","&general","ev_max_it","10");
  }else{  ##"'NC'"
     set_entry("$parfile","&general","nc_max_it","10");
  }
}


sub eff_initialize {
## initializes efficiency scan:
## determines possible n_procs_sim for the problem and stores them in global @n_procs_list
## also returns that list as a string
## in case a valid n_procs_sim is found in scan namelist,
##  @n_procs_lists will only contain this value
  my $nscans = shift;
  my $list_string = "";
  if ($efficiency == 1){
     printf "############################################\n";
     printf "# --efficiency option: max num procs = %3d #\n",$n_pes ;
     printf "#                      min num procs = %3d #\n",$min_procs ;
     if ($procs_per_node>0){
     printf "#                      procs p. node = %3d #\n",$procs_per_node ;
     }
     printf "############################################\n";
  }

# create output file efficiency.log
  if ($continue_scan==0){
     open(ELOG,">$SCANDIR/efficiency.log");
     my $entry = "";
     if ("$comp_type" eq "'EV'"){
        $entry = "#runnum | n_procs_sim | time for EV it. | total cpu time for EV it.\n";
     }else{
        $entry = "#runnum | n_procs_sim | time per step | total cpu time per step\n";
     }
     printf ELOG $entry;
     close(ELOG);
  }
  my ($valid, $n_parallel_sims, $n_procs_sim) = get_valid_scan_namelist("$tmpparfile");
  if (($valid==1)&&($efficiency==0)) {
#    efficiency scan allready knows n_parallel_sims and n_procs_sim
#    if --efficiency option is set, this is not possible because scan namelist is deleted before
     $list_string = "$n_procs_sim";
     @n_procs_list = $list_string;
  }else{
#    determine possible n_procs_sim for our problem (combination of prime factors)

#    size of the problem
     my $n_spec =  ReadValue("$tmpparfile","n_spec",1,0);
     my $nx0 =     ReadValue("$tmpparfile","nx0",1,0);
#    x parllelization does not work in local code: efficiency.log should ignore resulting errors
     my $nky0 =    ReadValue("$tmpparfile","nky0",1,0);
     my $nz0 =     ReadValue("$tmpparfile","nz0",1,0);
     my $nv0 =     ReadValue("$tmpparfile","nv0",1,0);
     my $nw0 =     ReadValue("$tmpparfile","nw0",1,0);
     my $x_local = ReadValue("$tmpparfile","x_local",1,0);
     my @numbers;  #phase space directions considered for parallelization
     if ($x_local == /f|F/ ){
         @numbers=($n_spec,$nw0,$nv0,$nx0,$nky0,$nz0);
     }else{
         @numbers=($n_spec,$nw0,$nv0,$nky0,$nz0);
     }
     expsinit();
     foreach (@numbers){
       @exps=prime_factors($_);
       for (my $i=0;$i<$max_num_primes;$i++){
         @all_exps[$i]+=@exps[$i];
       }
     }
     #now we have counted exponents of prime factors
     expsinit();
     traver2(0,$nscans); # modifies global @n_procs_list
     #print "n_procs_list\n";
     @n_procs_list = sort{$b<=>$a}(@n_procs_list);
     $list_string = join(",",@n_procs_list);
  }
  if ("$list_string" ne ""){
     print "   performance will be tested for n_procs_sim = ".$list_string."\n";
     return $list_string;
     #this list should now be printed into the scanstruct all other scans are to be ignored
  }else{
     print "no valid parallelization found -- exit!\n";
     print "criteria:\n";
     print "1) integer n_parallel_sims = n_pes/n_procs_sim\n";
     print "2) n_parallel_sims <= number of scans\n";
     print "3) integer n_procs_sim/procs_per_node  (minimizes inter-node communication)\n";
     print "4) n_procs_sim*n_parallel_sims>= n_pes (no idle processes)\n";
     exit(0);
   }
}

sub expsinit{ #initializes the global array @exps with zeros
  @exps = 0;
  for (my $count=1;$count<$max_num_primes;$count++){
     push(@exps,0); #initialize exponents
  }
}

sub traver2 { # recursively multiplies prime factor combinations
  my $i = shift ; #depth of recursion
  my $nscans = shift;
  my $n_procs = 0;
  for (my $j=0;$j<=$all_exps[$i];$j++){
     @exps[$i]=$j; #this instance of traver2 goes through exps[i])
     if ($i<$max_num_primes){
        traver2($i+1,$nscans);
     }else{
        $n_procs=compute_n_procs();
        add2list($n_procs,$nscans);
    }
  }
}

sub compute_n_procs { #computes n_procs frome exponents array
  my $res=1;
  for (my $i = 0;$i<$max_num_primes;$i++){
     $res*=@primes[$i]**@exps[$i];
  }
return $res ;
}

sub add2list{ #adds a entry to the @n_procs_list list
  my $element = shift ;
  my $nscans = shift ;
  my $inlist = 0;
  my $not_add_reason = -1;                # only add element if not_add_reason==-1
  if ($procs_per_node>0){                 # if procs_per_node is given
     my $n1 = $element/$procs_per_node;   # -fill whole node
     if (not isint($n1)){ $not_add_reason = 1;}
  }
  foreach (@n_procs_list){
     if ($_ == $element){$inlist = 1;}
  }
  if ($inlist==1){$not_add_reason = 2;}
  if ($element>$n_pes){$not_add_reason = 3;}  #does not fit
  if ($element<$min_procs){$not_add_reason = 4;}
  if ($max_n_par_sims<floor($n_pes/$element)){$not_add_reason = 5;}
  #if ($nscans*$element<=$n_pes){$not_add_reason = 5;}  #idle procs (moved to min_procs)
  if (($efficiency==0)&&(not(isint($n_pes/$element)))){
     #integer n_parallel_sims desired
     $not_add_reason = 6;
  }
  if ($not_add_reason==-1){
     push(@n_procs_list,$element);
  }
  if ($verbose==1){
     if ($not_add_reason!=-1){print "n_procs_sim=$element not valid:\nreason ";}
     if ($not_add_reason==1) {
        print "1) does not fill whole nodes (ppn=$procs_per_node)\n";
     }elsif ($not_add_reason==2) {
        print "2) is in list\n";
     }elsif ($not_add_reason==3) {
        print "3) is larger than n_pes=$n_pes\n";
     }elsif ($not_add_reason==4) {
        print "4) leaves idle processors: not enough scans ($nscans)\n";
     }elsif ($not_add_reason==5) {
        print "5) required n_parallel_sims larger than max_n_par_sims\n";
     }elsif ($not_add_reason==6) {
        print "6) n_pes/n_procs_sim not integer\n";
     }
  }
}

sub prime_factors { # finds exponents of prime factors of a number
  my $num = shift;
  my @exponents ;
     for (my $i=0;$i<$max_num_primes;$i++){
        my $prime = @primes[$i];
        while(isint((1.0*$num)/(1.0*$prime))){
           my $x = ($num/$prime);
           $num=(1.0*$num)/(1.0*$prime);
           @exponents[$i]++;
        }
     }
  #no prime factors > 23 are considered for processor numbers
  #but a grid size of a prime number >23 (e.g. 31) remains allowed
  return @exponents ;
}

sub write_efflog{
  my $filenum = shift;
  my $parfile = sprintf("$SCANDIR/parameters_%.4d",$filenum);
  my $my_procs = ReadValue($parfile,"n_procs_sim",1,0);
  my $step_time = 0;
  if ("$comp_type" eq "'IV'"){
     $step_time = ReadValue($parfile,"step_time",1,0); # time per time step
  }elsif("$comp_type" eq "'EV'"){
     $step_time = ReadValue($parfile,"time for eigenvalue solver",1,0); # time for Eigenvalue solver
  }elsif("$comp_type" eq "'NC'"){
     $step_time = ReadValue($parfile,"time for nc equilibrium solver",1,0); # time for Eigenvalue solver
  }else{
     #unknown comp_type
     $step_time=0
  }

  my $cpu_time = $step_time * $my_procs; #cpu time per timestep
  my $entry = sprintf("%.4d    | ",$runnum);
  my $nan=0;
  if ($my_procs ne 0){
     $entry="$entry".sprintf("%.4d        | ",$my_procs) ;
  }else{
     $entry="$entry".sprintf("NaN         | ");
     $nan=1;
  }

  if ($step_time != 0){
     $entry=sprintf("$entry"."%.6f      | %.6f\n",$step_time,$cpu_time) ;
  }else{
     $entry=sprintf("$entry"."NaN           | NaN    \n");
     $nan=1;
  }
  if ($nan eq 1) {
     print "some error occured in routine write_efflog, skip entry $runnum!\n";
  }else{
     open(ELOG,">>$SCANDIR/efficiency.log");
     printf ELOG $entry;
     close(ELOG);
  }
}
####### END OF ROUTINES FOR THE --efficiency OPTION ###################



####### ROUTINES CONCERNING AUTOPARALLEILIZATION #########################

sub get_autopar_condition {
    # cond = 0   --> if autoparallelization is done, it is done for every run
    # cond = 1   --> if autoparallelization is done, it is done only once
    # check if  (1)scan over resolution parameters (nkx0....)
    #                  (2)scan over "beta" or "coll" including "0"
    #                   because gene uses different routines for beta/coll =0 or !=0

    #--------------resolution parameters-------------------------------------#
    my $filename = shift;
    my $ap_switch = shift;   # first run: autoparallelization handling on (ap_switch=1) (later: not necessary)
    my $ap_box_cond = 1;  #assume the resolution is not changed
    my $ap_bc_cond = 1; # beta coll condition assume: no problems..
    my $ap_other_cond = 1; # bc and nblocks combined
    my $ap_perf_cond = 1; #assume nblocks and perf_vec is not fixed by the user
    if ($efficiency==0){
      open(FH,"<","$filename");# || die "couldn't open dir.";
      print "   check if autopar settings can be taken from efficiency test runs...\n";
      if($ap_switch==0){
         print "   ->no (user setting --ap_switch $ap_switch)\n";
      }
      while (<FH>) {
         if ($_ =~ /\s*(\w+)\s*=\s*(\S+)\s*\!scan(\w*):/){
           if(("$1" eq "nx0")
             or ("$1" eq "nky0")      or ("$1" eq "nz0")
             or ("$1" eq "nw0")      or ("$1" eq "nv0")
             or ("$1" eq "n_spec")   or ("$1" eq "n_procs_sim")) {
                $ap_box_cond=0; #yes, the box or n_spec is changed
                print "   ->no (!scan in $1)\n";
           }
         }
      }
      close(FH);
    }else{
      $ap_box_cond=0;
        print "   !scan in n_procs_sim (--efficiency): do autoparallelization for every run\n";
    }

    #--------------beta coll condition -------------------------------------#
    # go through all scan values and check if one beta or coll is 0
    my $nosteps = 0;
    my $endval = 0;
    my $formula = "";
    my $val = 0;
    my $startval = 0;
    my $step = 0;
    my $i = 0 ;    #number of scan parameter in the struct
    my $name = "";
    my $scantype = "0";
    my @list = "";
    while (defined $StrInt->name($i)){
      my $fktindex = 1; #set to 1 before each entry
      $name = $StrInt->name($i);
      if ("$name" =~ /collision_op/){$ap_bc_cond=0;}
      if (("$name" =~ /beta/) or ("$name" =~ /coll/)){
         $scantype = $StrInt->scantype($i);
         if ("$scantype" eq "list"){
            @list=split(/,/,$StrInt->step($i));
            foreach $val(@list){
               if ($val-0.==0.){
                  $ap_bc_cond=0;
                  print "   ->no ($name becomes 0 in scan)\n";
               }
            }
         }elsif ("$scantype" eq "func") {
            $nosteps  = $StrInt->start($i);
            $formula = $StrInt->step($i);
            $endval     = $StrInt->end($i);
            $val = function(1,$formula);
            while (($val<$endval)&&($fktindex<=$nosteps)){
               $val = function($fktindex,$formula);
               if ($val-0. == 0.) {
                  $ap_bc_cond = 0;
                  print "   ->no ($name becomes 0 in scan)\n";
               }
               $fktindex++;
            }
         }else{ #some range..
            $startval  = $StrInt->start($i);
            $step = $StrInt->step($i);
            $endval     = $StrInt->end($i);
            $val = $startval;
            while (($step*$val <= $step*$endval) && ($step != 0.) ){
               if ($val-0. == 0.){
                  $ap_bc_cond = 0;
                  print "   ->no ($name becomes 0 in scan)\n";
               }
               $val = $val + $step;
            }
         }
      }
      $i++;
   }
   ## if perf_vec or nblocks given in the parameter file-> use it in all runs of the scan
   if (read_entry($filename,"nblocks") ne ""){
      $ap_perf_cond = 0;
      print "   -nblocks is given in the parameters file: use it in autoparallelization for every run\n";
   }
   if (read_entry($filename,"perf_vec") ne ""){
      $ap_perf_cond = 0;
         print "   -perf_vec is given in the parameters file: use it in autoparallelization for every run\n";
   }
   $ap_other_cond=$ap_bc_cond*$ap_perf_cond;
   #--------------------return all conditions in an array---------------------------#
   my @autopar_cond =  ($ap_box_cond,$ap_other_cond,$ap_switch);
   if ($ap_box_cond*$ap_other_cond*$ap_switch==1){
      print "   ->yes\n";
   }
   if ($verbose==1){
      print "   (conditions: box $ap_box_cond, beta/coll $ap_bc_cond, perf $ap_perf_cond, switch $ap_switch)\n";
   }
   print "\n";
   return @autopar_cond ;
}
####### END OF ROUTINES CONCERNING AUTOPARALLEILIZATION #########################

##################################################################################
sub prepare_scandir {
  #creates the output directories
  #sets the paths in parameter files
  my $i=0;
  my $scanfiles=" ";
  my $outdir_orig=$OUTDIR;
  if ($OUTDIR=~" "){
     $OUTDIR=ReadValue("$PROBDIR/parameters","diagdir",1,0);
     $OUTDIR=~/\'(\S+?)\/?((scanfiles)\d*)?\/?\'/;
     $OUTDIR=($1);  #cut the /scanfilesXXX part from OUTDIR
     $scanfiles = ($2);
  }else{
     #$/="/";
     #chomp($OUTDIR);
     $OUTDIR=~/\'(\S+?)\/?((scanfiles)\d*)?\'/;
     $OUTDIR=($1);
     $scanfiles = ($2);
  }
  print "   using PROBDIR = $PROBDIR\n";
  if(not (-d "$OUTDIR") ) {
     print "   OUTDIR does not exist -- try to create $OUTDIR\n";
     mkdir("$OUTDIR");
     if(not (-d "$OUTDIR") ) {
       print "   could not create $OUTDIR, using $outdir_orig instead\n";
       $OUTDIR=$outdir_orig;
     }
  }
  #check OUTDIR
  opendir(OUTDIR,"$OUTDIR") || die "couldn't open dir: $OUTDIR\n";
  closedir(OUTDIR);
  chdir("$PROBDIR");
  if($continue_scan==0){
     print "   creating new scandir.\n";
     $i=0;
     $SCANDIR=sprintf("%s/%s%.4d",$OUTDIR,"scanfiles",$i);
     # tests if scanfiles exist
     while(-e "$SCANDIR"){
        $i++;
        $SCANDIR=sprintf("%s/%s%.4d",$OUTDIR,"scanfiles",$i);
     }
     # avoid that two simultaneously starting scans choose the same
     # (file system delay) by waiting a short random time
     sleep_random("$PROBDIR");
     while(-e "$SCANDIR"){
        $i++;
        $SCANDIR=sprintf("%s/%s%.4d",$OUTDIR,"scanfiles",$i);
     }
     mkdir("$SCANDIR");
  }else{
     #use the last scanfile
     $SCANDIR="$OUTDIR/$scanfiles";
     if(not(-e "$SCANDIR")){
        print "failed to open SCANDIR:\n$SCANDIR\n--exit\n";
        exit()
     }
  }
  print "   using SCANDIR = $SCANDIR\n";
  $PARINDIR="$SCANDIR/in_par";
  print "   using PARINDIR = $PARINDIR\n";
  if(-e ("$PROBDIR/parameters") ) { #regular case
     ChEntry("$PROBDIR/parameters","diagdir","\'$SCANDIR\'\n",1,"silent");
     ChEntry("$PROBDIR/parameters","chptdir","\'$SCANDIR\'\n",1,"silent");
     #original parameters copied to scandir
     copy("parameters","$SCANDIR/parameters");
  }else{ #otherwise, assume we're in the scandir already with a copied parfile
     $PROBDIR=$SCANDIR;
     $tmpparfile="$SCANDIR/tmp_parameters";
  }
  #create temporary parameters file
  copy("$PROBDIR/parameters","$tmpparfile");

  set_parindir("$tmpparfile");
  set_parindir("$PROBDIR/parameters");
  set_parindir("$SCANDIR/parameters");

} #end prepare_scandir


###########################################################################


sub create_scanlog{
  # creates scan.log, writing the header line
  # creates neo.log, writing the header line
  my $entry = "";
  my $i=0;
  my $n_scanpars=@scancoords;
  for (my $i=0;$i<$n_scanpars;$i++){
       $entry= sprintf("| %-9s %d  %s",$StrInt->name($i),$StrInt->spec($i),$entry);
  }
  $entry="#Run  ".$entry;
  open(LOG,">$SCANDIR/scan.log");
  print LOG $entry;
  for(my $i=1;$i<=$n_ev;$i++){
    print LOG "/Eigenvalue$i            ";
  }
  print LOG "\n";
  close(LOG);

  #### neoclassical fluxes
  my $istep_neoclass = read_entry("$tmpparfile","istep_neoclass");
  my $n_spec = read_entry("$tmpparfile","n_spec");
  if ($istep_neoclass eq "") {$istep_neoclass = 0}
  if (($istep_neoclass>0) or ("$comp_type"eq"'NC'")){
    open(NEOLOG,">$SCANDIR/neo.log");
    print NEOLOG $entry;
    for(my $i=1;$i<=$n_spec;$i++){
        print NEOLOG "/  G_neo$i    Q_neo$i    Pi_neo$i   j_b$i      ";
    }
    print NEOLOG "\n";
  close(NEOLOG);
  }
  return();
}

###########################ROUTINES CONCERNING THE SCAN STRUCT ########################

sub read_parameters{
  # parse parameters file for scans
  # fills global struct 'StrInt' containing scan information
  # initializes global array 'scan_coords' with the number of scan dimensions
  # sets parameter values in input 'parfile' to first values of scan
  # returns total number of scan values
  my $parfile = shift;
  my $silent = shift;
  my @my_scancoords;
  my $i=0;
  my $j=0;
  my $k=0;
  my $line;
  my $entry="";
  my $name = "";
  my $scantype = "";
  my $cmpl = "";
  my $values = "";
  my $firstval = 0;

  if ("$silent" ne "silent"){
     print "   reading parameters (setting first scan values):\n";
  }
  open(INTFILE,"<$tmpparfile");
  my @datarrs=<INTFILE>;
  close(INTFILE);

  initialize_StrInt();

  for($k=0;$k<=$#datarrs;$k++){
     $line=$datarrs[$k];
     if ($line=~/\s*(\S+)\s*=\s*.+\s*!scan(\w*):((re|im):)?\s*(.+,.+)\s*/){
        # $values has to have "," --> functional dependencies work, one-element lists are ignored!!
        $name = $1;
        $scantype = $2;
        # $3 is $complex with ":" ...
        if (defined $4){$cmpl = $4;}
        $values = $5;
        chomp($values);
        #BACKWARDS COMPATIBILITY
        if ("$scantype" eq ""){
           if ($values=~/xi/){
              $scantype="func";
           }else{
              $scantype="range";
           }
        }
        $StrInt->name($i,"$name");
        $StrInt->scantype($i,"$scantype");
        $StrInt->spec($i,1);
        for ($j=0;$j<$k;$j++){
           if ($datarrs[$j]=~/^\s*$name\s*=/){
              $StrInt->spec($i,$StrInt->spec($i)+1);
           }
        }

        # scan-type dependent struct filling:
        if ("$scantype" eq "range"){
           my @varlist =  split(/,/,$values);
           #note: testing isnumber($varlist[0]) lead to frequent problems
           my $len=@varlist;
           if ( $len==3 ) {
              $StrInt->start($i,$varlist[0]);
              $StrInt->step($i,$varlist[1]);
              $StrInt->end($i,$varlist[2]);
              $firstval = $varlist[0];
           }else{
              print "$line\n";
              print "invalid scan$scantype syntax in line \n $line\n need 3 comma separated numbers -- exit!\n";
              exit(0);
           }
        }elsif ("$scantype" eq "func"){
           #my @varlist =  split(/,/,$values);
           #my $len=@varlist;
           #if ( $len==3 ) {
           $values=~/\s*(.+),(.+),(.+)\s*/;
           #test: $1 and $3 must be numbers $2 is some formula not tested
           if (( defined $1) and (defined $2) and (defined $3)){
              $StrInt->start($i,$1);
              $StrInt->step($i,$2);
              $StrInt->end($i,$3);
              $firstval = function(1,$2);
           }else{
              print "$line\n";
              print "invalid scan$scantype syntax in line \n $line\n";
              print "need 3 comma separated entries -- exit!\n";
              exit(0);
           }
        }elsif ("$scantype" eq "list"){
           #list without test, $values is saved as a string
           $StrInt->start($i,0);
           $StrInt->step($i,$values);      #save the string, not the array.
           $StrInt->end($i,0);
           $firstval = (split(/,/,$values))[0];
           $firstval = function(1,$firstval);
	}elsif ("$scantype" eq "with"){
           #list without test, $values is saved as a string
	   $values=~/\s*(.{1}),(.+)\s*/;
	   if ((defined $1) and (defined $2)){
              $StrInt->start($i,$1); #fill start with index or name of tracked parameter
	                             #in the latter case, start will be overwritten in SolveDep
              $StrInt->step($i,$2);  #save the string, not the array.
              $StrInt->end($i,0);
              $firstval = (split(/,/,$2))[0];
              $firstval = function(1,$firstval);
	   }else{
	      print "$line\n";
	      print "invalid scan$scantype syntax. \n";
	      print "need comma separated values, where the first gives the index of variable to scan with.\n";
	      exit(0);
	   }
        }else{
           print "unknown scan expression !scan$scantype occured at parameter $name\n";
           exit(0);
	}
        # complex values are written by ChEntry in a special way
        if (("$cmpl" eq "re") or ("$cmpl" eq "im")){
          $StrInt->name($i,"$name"."_scan_$cmpl");
        }
        # set firstval
        ChEntry("$tmpparfile",$StrInt->name($i),"$firstval",$StrInt->spec($i),"$silent");
        $i++;
        push(@my_scancoords,0); #the lenght of this is the number of scan parameters
     }
  }

  # no scan found for i=0
  if ($i == 0) {
     print "\n### no scan was found in the parameters file ###\n";
     print   "### set scan_dims = 1                        ###\n";
     push(@my_scancoords,0);
  }

  #for setting first values:
  SolveDep("$tmpparfile","$silent");
  if ("$silent" ne "silent"){print "\n";}

  @scancoords = @my_scancoords; #initialization...
  return($i);
}# end parse_parameters

###########################ROUTINES CONCERNING THE SCAN STRUCT ########################


###########################RECURSIVELY PARSING THE SCAN STRUCT ########################

sub Traver{
  # traverse scan struct recursively (see comments at end of file)
  # what actually is done depends on the 'action' input
  my $i = shift;         #depth of recursion (numbers scan parameters), start with 0
  my $max_scans = shift; #maximal depth of recursion (avoids uncontrollable number of recursions at error)
  my $action = shift;    #what Traver should do
  my $silent = shift;    #if "silent", no output!
  my $ap_box_cond = $autopar_cond[0];
  my $val=0.;
  my $scanindex=1; #count the values that parameter $i takes
  my $listlength=0;
  my @varlist = "";
  ## START VALUE
  if ($StrInt->scantype($i) eq "func" ){
     #here start with index 1
     $val=function($scanindex, $StrInt->step($i));
  }elsif($StrInt->scantype($i) eq "list"){
     @varlist= split(/,/,$StrInt->step($i));
     $listlength = @varlist ;   # determine length
     $val=$varlist[$scanindex-1]; #here start with index 0
     $val = function(1,$val);
  }elsif($StrInt->scantype($i) eq "with"){
     @varlist= split(/,/,$StrInt->step($i));
     $val=$varlist[$scancoords[$StrInt->start($i)]-1];
     $val = function(1,$val);
  }else{
     $val=$StrInt->start($i);
  }

  my $whilecond;
  my $val1=$val;
  my $entryname = "";
  do{
     $entryname=$StrInt->name($i);
     ChEntry("$tmpparfile",$entryname,"$val",$StrInt->spec($i),"$silent");
     if (("$val" =~ m/[a-z,A-Z]/) and ( "$val" !~ m/([Ee][+-])/ )) {
        #$val is a string (this might not work with diagnostics...but its nice!)
        $datarr[$i]=sprintf(" | %12s",$val);
     }else{
        #here val should be a number
        $datarr[$i]=sprintf(" | %12.6e",$val);
     }
     $scancoords[$i]=$scanindex;   #write (karthesian) scan coordinates to array
                                   #overwrite until maximum value in dimension i is reached
     if (($multilist==1) && ($i>0)){
        $scancoords[$i]=1;         #for multilist function only the first parameter sets dimension
     }
     $StrInt->scanindex($i,$scanindex); #count index of each scan variable

     if ((defined $StrInt->name($i+1))and($i<$max_scans)){
        # Traver recursion : call next Traver, until last par in struct
        # then, run gene instead and go one level up in recursion
        Traver($i+1,$max_scans,"$action","$silent");
     }else{
        #  treat valid scan parameter combinations only
        my $is_valid_run=decide_if_valid_run($runnum,$max_scans);
        if ($is_valid_run==1) {

           SolveDep("$tmpparfile","$silent");

           #calc runnum
           $runnum += 1;
           if ("$silent"ne"silent") {
              printf "   ...run number %.4d \n",$runnum;
           }

           ## TRAVER MULTIFUNCTIONALITY: DIFFERENT CASES ##########33
           if ("$action" eq "writepar"){
              #CASE 1 only copy parameters to $PARINDIR
              my $newfilename = sprintf("$PARINDIR/parameters_%.4d",$runnum);
              copy("$tmpparfile","$newfilename");
              if ($reuse_geom != 0){
                 my $maggeom = ReadValue("$PROBDIR/parameters","magn_geometry",1,0);
                 $maggeom=~/\'(\S+?)\'/;
                 $maggeom=($1);
                 my $n_parallel_sims = ReadValue("$tmpparfile","n_parallel_sims",1,0);
                 my $geomfile = sprintf("%s_%.4d",$maggeom,1);
                 if(($runnum > $n_parallel_sims) or (-f "$SCANDIR/$geomfile")){
                     set_entry("$newfilename","&geometry","magn_geometry","'gene'");
                     set_entry("$newfilename","&geometry","geomdir","'$SCANDIR'");
                     set_entry("$newfilename","&geometry","geomfile","'$geomfile'");
                     print("edit $tmpparfile to use geomfile $geomfile $n_parallel_sims\n");
                 }
              }
           }elsif ("$action" eq "countscan"){
              #only parse, no 'action'...
           }elsif("$action" eq "scanlog"){
              #CASE 3 parse and create scan.log
              process_output_data($i);
           }else{
              print "unknown Traver action\n";
           }
        }
     }  #this belongs to "if i+1 entry in struct does not exist..

     # STOP CONDITION and SETTING OF NEXT $val
     $val1 = $val;
     $scanindex++;
     if ($StrInt->scantype($i) eq "func"){
        if ($scanindex <= $StrInt->start($i)){
           $val=function($scanindex, $StrInt->step($i));
           $whilecond=($StrInt->end($i)-$val)*10000000*($val-$val1); ;
        }else{
           $whilecond=-1;
        }
     }elsif ($StrInt->scantype($i) eq "list"){
        if ($scanindex <= $listlength){ # scanindex runs  1..length
           $val=$varlist[$scanindex-1];  # list index runs 0..length-1
           $val=function(1,$val);
        }else{
           $whilecond=-1;
        }
     }elsif ($StrInt->scantype($i) eq "with"){
        $whilecond=-1;
     }elsif ($StrInt->scantype($i) eq "none"){
        $whilecond = -1;
     }else{
        $val+=$StrInt->step($i);
        $whilecond=($StrInt->end($i)-$val)*10000000*($StrInt->step($i));
     }
  }while($whilecond > -0.00000001);
  return 0;
}

##################################################################################

sub decide_if_valid_run {
  my $runnum=shift;
  my $max_scans=shift;
  my $isvalidscan=1;
  my $i=0;
  my $i0=$StrInt->scanindex(0);
  if ($multilist==1) {
     while ((defined $StrInt->name($i))){
        if ($StrInt->scanindex($i) != $i0){
           $isvalidscan=0;
        }
        $i++;
     }
  }
  if ((($runnum+1) <= $max_scans) && ($isvalidscan==1)){
     $isvalidscan=1;
  }else{
     $isvalidscan=0;
  }
  return $isvalidscan;
}

sub execute_gene{
  #executes the GENE code and re-directs output
  #np_gene is the number of MPI processes addressed
  my $np_gene = shift;
  my $tmpgerr = "$SCANDIR/tmpgene.err";
  my $syscall = "";
  $syscall = "$make -j -f /scratch/project_462000451/gene_enchanted/makefile run N_PES=$np_gene NLSPATH=$ENV{'NLSPATH'}";
  printf "\n   GENE CALL, n_procs = %.4d         \n\n",$np_gene;

  if ($mysyscall ne "") {
     $syscall="$mysyscall";
  }
  $syscall.=" 2>&1 1>&1 | tee $tmpgerr";

  if ($stop==1) {
     print "stop script before executing\n$syscall\n";
     exit(0);
  }

  my $parent_pid = $$;
  #  gene output is monitored for errors
  #  in case of error: stop scan
  my $child_pid = fork();
  if($child_pid){
     #parent process (scanscript?) has to wait until child(gene?) is ready
     wait;
  }else{
     ##gene call
     if ((qx($syscall))=~/(STOP.*|ERROR.*)/) {
        print "\nProblem with compiling/running gene. For details see:\n $tmpgerr\n. Last ERROR Message:\n  ($1)\n";
        #empty eigenvalues.dat because of wrong eigenvalues
        if($1=~/(PETSC ERROR:.*|PETSC WARNING:.*)/){
           open(EIGENS,">$OUTDIR/eigenvalues.dat");
           close(EIGENS);
           exit(0);
        }
        if ($force==0){
           # option --force : continue scan if errors occur.
           exit(getppid());
           exit(0);
        }
     }
     exit(0);
  }
}

sub process_output_data   {
  #print "process output data...\n";
  my $i=shift;  # $i is the maximal recursion depth reached
  my $nky0 = ReadValue("$tmpparfile","nky0",1,0);
  my @printarr;
  my $j;
  my $line;
  my @entry;
  my $filenum=$runnum;
  my $newfilename=' ';

  rename_output_files($filenum);   #in the scandir

  my $scanparfile=sprintf("$SCANDIR/parameters_%.4d",$filenum);
  my $inparfile=sprintf("$PARINDIR/parameters_%.4d",$filenum);

  for($j=$i;$j>=0;$j--){  #$i is max recursion depth here
     if($StrInt->name($j) eq "kymin"){
        #replace ky string if GENE adapted kymin
        my $kymin_in=sprintf("%12.6e",ReadValue("$inparfile","kymin",1,0));
        my $kymin_out=sprintf("%12.6e",ReadValue("$scanparfile","kymin",1,0));
        if (($kymin_in!=$kymin_out)&&(($nky0==1)||("$comp_type" eq "'EV'"))) {
           $datarr[$j] =~ s/$kymin_in/$kymin_out/;
        }
     }
  }

  # write scan.log entry (both for comp_type='IV' or  'EV')
  @printarr=@datarr;
  my $omega_prec = ReadValue($scanparfile,"omega_prec",1,0);
  my $ndigits=4;
  my $nanfmt=' | %7s %7s';
  my $fmt=' | %7.4f %7.4f';
  if ($omega_prec!=0){
     $ndigits=floor(log10(1.0/$omega_prec));
     #if ($ndigits<4){ $ndigits=4; }
     $nanfmt=sprintf(' | %%%ds %%%ds',$ndigits+3,$ndigits+3);
     $fmt=sprintf(' | %%%d.%df %%%d.%df',$ndigits+3,$ndigits,$ndigits+3,$ndigits);
  }
  if ("$comp_type" eq "'IV'"){
     my $scan_om_file = sprintf("$SCANDIR/omega_%.4d",$filenum);
     open(FILE,"<$scan_om_file");
     @entry=<FILE>;
     close(FILE);
     if ((-z $scan_om_file)or(not -e $scan_om_file)){
        $printarr[0].=sprintf($nanfmt,"NaN","NaN");
     }else{
        foreach $line(@entry){
           if($line=~/\s*(-?\S+)\s+(-?\S+)\s+(-?\S+)\s*/){
              if (($2==0)&&($3==0)){
                 $printarr[0].=sprintf($nanfmt,"NaN","NaN");
              }else{
                 $printarr[0].=sprintf($fmt,$2,$3);
              }
           }else{
              # no match...
              $printarr[0].=sprintf($nanfmt,"NaN","NaN");
           }
        }
     }
  }else{  ## $comp_type!= 'IV'
     my $scanEVfile = sprintf("$SCANDIR/eigenvalues_%.4d",$filenum);
     open(EIGENS,"<$scanEVfile");
     @entry=<EIGENS>;
     close(EIGENS);
     #chomp(@entry);
     $j=-1;
     foreach $line(@entry){
        if ($j<0){
           if ($line=~/eigenvalues/){
              $j++;
           }
        }else{
           $line=~/\s*(-?\S+)\s+(-?\S+)\s*/;
           $printarr[0].=sprintf($fmt,$1,$2);
           $j++;
        }
     }
     while($n_ev-$j>0){
        $printarr[0].=sprintf($nanfmt,"NaN","NaN");
        $j++;
     }
  }
  # write scanlog
  open(LOG,"<$SCANDIR/scan.log");
  my @scanentry=<LOG>;
  for($j=$i;$j>=0;$j--){
     $scanentry[$runnum].=$printarr[$j];
  }
  $scanentry[$runnum]=sprintf("%.4d %s\n",$runnum,$scanentry[$runnum]);
  close LOG;
  open(LOG,">$SCANDIR/scan.log");
  for($j=0;$j<=$#scanentry;$j++){
      print LOG $scanentry[$j];
  }
  close(LOG);
  write_neolog($filenum,$i);
}

####################################################################################

sub write_neolog {
  my $filenum = shift;
  my $i = shift;
  my $line = "";
  my @entry;
  my $ncentry = "";
  my $istep_neoclass = read_entry("$tmpparfile","istep_neoclass");
  if ($istep_neoclass eq "") {$istep_neoclass = 0;}
  my $n_spec = read_entry("$tmpparfile","n_spec");
  my $n = 1;
  my @Gamma ;
  my @Qheat ;
  my @Pmmtm ;
  my @jboot ;
  my @specname;
  if (($istep_neoclass>0)or("$comp_type"eq"'NC'")){
     my $neo_file = sprintf("$SCANDIR/neoclass_%.4d",$filenum);
     if (not(-e $neo_file)){
        while($n<=$n_spec) {
           $ncentry=$ncentry." NaN       NaN       NaN       NaN     ";
           if($n<$n_spec){$ncentry = $ncentry."  | ";}
           $n++
        }
     }else{
        #read the last line for each species in backwards species order
        tie *BW, 'File::ReadBackwards', "$neo_file" or die "can't read 'neo_file' $neo_file" ;
        while( ($line=<BW>) and ($n<=$n_spec)) {
           if ($line  =~/\s*(-?\S+)\s+(-?\S+)\s+(-?\S+)\s+(-?\S+)\s*/){
              $Gamma[$n-1] = $1;
              $Qheat[$n-1] = $2;
              $Pmmtm[$n-1] = $3;
              $jboot[$n-1] = $4;
           }
           #print "for species $n_spec-$n+1 found\n $line\n";
           $n++;
        }
        # write in species order of the parameters file
        for($n=1;$n<=$n_spec;$n++){
           if (($Gamma[$n_spec-$n]==0)&&($Qheat[$n_spec-$n]==0)&&($jboot[$n_spec-$n]==0)){
              $ncentry=$ncentry." NaN       NaN       NaN       NaN     ";
           }else{
              $ncentry=$ncentry.sprintf("%9.6f %9.6f %9.6f %9.6f",$Gamma[$n_spec-$n],$Qheat[$n_spec-$n],$Pmmtm[$n_spec-$n],$jboot[$n_spec-$n]);
           }
           if($n<$n_spec){$ncentry = $ncentry."  | ";}
        }
     }
     #write values of the scan parameters ($i is at maximum recursion depth)
     my $parvals = "";
     for(my $j=0;$j<=$i;$j++){
       $parvals=$datarr[$j].$parvals;
     }
     $ncentry=sprintf("%.4d ",$filenum).$parvals." | ".$ncentry."\n";
     open(NEOLOG,">>$SCANDIR/neo.log");
     print NEOLOG $ncentry;
     close(NEOLOG);
  }
}

####################################################################################
sub create_scan_namelist{
## add a scannamelist to parameter $file, if it does not exist
  my $file = shift;
  my $line;
## if there is no &scan namelist: create it
#  !&scan does not count!
  my $scan_exists=0;
  open(FH,"<$file");
  while ($line=<FH>) {
    if ($line =~ /^\s*&scan/) { $scan_exists=1; }
  }
  close(FH);
  if ($scan_exists ==0){
    open(FH,">>",$file);
    printf FH "\n&scan\n/\n";
    close FH;
    #print "\nscan namelist is created in $file\n"  ;
  }
}

sub edit_scan_namelist{
## sets the entries of the scan namelist of parameter $file
  my $file = shift;
  my $n_parallel_sims = shift;
  my $n_procs_sim = shift;

  create_scan_namelist("$file");  #if it does not exist...

#   set all entries
    set_entry("$file","&parallelization","n_parallel_sims", "$n_parallel_sims");
    set_entry("$file","&parallelization","n_procs_sim", "$n_procs_sim");
    set_entry("$file","&scan","par_in_dir", "'$PARINDIR'");
#   for the efficiency scan...
    set_entry("$file","&scan","scan_dims", "$n_parallel_sims");

}


sub delete_scan_namelist{
  my $file = shift;
  my $line;
  print "\ndelete entries of the scan namelist!\n"  ;
  delete_entry("$file","n_parallel_sims");
  delete_entry("$file","n_procs_sim");
  delete_entry("$file","scan_dims");
  delete_entry("$file","par_in_dir");
}



sub rename_output_files {
  my $filenum = shift;
  my $elem = "";
  opendir(PATH,"$SCANDIR");
  my @entry=readdir(PATH);
  closedir(PATH);
  foreach $elem(@entry){
     if($elem=~/tmpgene\.err/){
        open(ERRDAT,">>$SCANDIR/geneerr.log");
        print ERRDAT "\n run $filenum:\n";
        open(TMPERRDAT,"<$SCANDIR/$elem");
        my @errentry = <TMPERRDAT>;
        close(TMPERRDAT);
        print ERRDAT @errentry;
        close(ERRDAT);
        unlink("$SCANDIR/$elem");
        my $errelem='';
        foreach $errelem(@errentry){
           if(($errelem=~/(.*STOP.*)/) and ($force==0)){
              exit(); #if force=0 only write first error then stop.
           }
        }
     }
  }
}
####################################################################################

sub function{
   # evaluates f(xi)
   # (also can evaluate formulas in lists..)
    no strict 'refs';
    my $fktindex=shift;
    my $formula=shift;
    my $ch;
    while($formula=~/(!?[a-z_]+)\((\d)\)/){
      $ch = ReadValue("$tmpparfile",$1,$2,0);
      $formula=~ s/(!?[a-z_]+)\((\d)\)/$ch/;
    }
    $formula=~ s/xi/$fktindex/gi;
    $formula="\$ch=".$formula;
    eval "$formula";
    return($ch);
}

####################################################################################

#################################################################################
# SolveDep evaluates the functional dependency of a <par1>
# on other parameters <pari>
# for every <par1> = <val> !scan:formula(<pari>(#))
sub SolveDep{
  no strict 'refs';
  my $parfile=shift;
  my $silent=shift;
  open(FILE,"$parfile");
  my @entryarr=<FILE>;
  close(FILE);
  my $line="";
  my $spec=1;
  my $i=0;
  my $j=0;
  my $ch;
  my $formula;
  my $par;
  my $name;
  my $string;
  my $lenstr;
  my $value=0;
  my @parlist; my @strarr;
  # find line with formula
  # WARNING: no digits in name:  (e.g. !scan: q0(1)*3  does not work!
  foreach $line(@entryarr){
     #-> why this {2} ???.. {2} to exclude .t. and .f. used only for read_chekpoint and treated later..
     if($line=~/\s*(\S+)\s*=\s*.+ !scan:.*!?[A-Za-z]{2}/ && (not $line=~/,/)){
        $line=~/\s*(\S+)\s*=\s*.+ !scan:(.+)/;
        $name=$1;
        $formula=$2;
        # find species-number of parameter of formula
        # search file for occurencies of $name at line start up to present line number $i
        # only at beginning of line -> comments including $name are allowed
        for ($j=0;$j<$i;$j++){
           if (($entryarr[$j]=~/^\s*$name\s*=/)&&!($entryarr[$j]=~/diagdir/)&&!($entryarr[$j]=~/chptdir/)){
              $spec++;
           }
        }
        # substitute params in formula
        while($formula=~/(!?[A-Z,a-z,_,0-9]+)\((\d)\)/){
           $ch = ReadValue("$parfile",$1,$2,0);
           $formula=~ s/(!?[A-Z,a-z_,0-9]+)\((\d)\)/$ch/;
         }
        $formula="\$ch="."$formula";
        eval "$formula";
        ChEntry("$parfile",$name,"$ch ",$spec,"$silent");
     }elsif ($line=~/\s*(\S+)\s*=\s*(\S+)\s*!replace:\s*(\S+)\s*/){
        $name=$1;
        $string=$2;
        $formula=$3;
        $string =~ s/(^\s*'|'\s*$)//g;  #trim
        @strarr=split('_',$string);
        #count spec
        for ($j=0;$j<$i;$j++){
           if (($entryarr[$j]=~/^\s*$name\s*=/)&&!($entryarr[$j]=~/diagdir/)&&!($entryarr[$j]=~/chptdir/)){
              $spec++;
           }
        }
        @parlist = split(',',$formula);
        $lenstr=@strarr; #length
        $j=@parlist;
        $j=$lenstr-$j;
        #substitute string array elements with list entries (formulas allowed)
        foreach $formula(@parlist){
           while($formula=~/(!?[A-Z,a-z,_,0-9]+)\((\d)\)/){
              $ch = ReadValue("$parfile",$1,$2,0);
              $formula=~ s/(!?[A-Z,a-z_,0-9]+)\((\d)\)/$ch/;
           }
           $formula="\$ch="."$formula";
           eval "$formula";
           @strarr[$j]=$ch;
           $j++;
        }
        #join and write
        $string="'".join('_',@strarr)."'";
        ChEntry("$parfile",$name,"$string ",$spec,"$silent");
     }elsif ($line=~/\s*(\S+)\s*=\s*(\S+)\s*!scanwith:\s*(\S+)\s*/){
	$name=$1;
	$value=$2;
	$formula=$3;
	# find species-number of parameter that has scanwith
	# search file for occurencies of $name at line start up to present line number $i
	# only at beginning of line -> comments including $name are allowed
	for ($j=0;$j<$i;$j++){
	    if (($entryarr[$j]=~/^\s*$name\s*=/)&&!($entryarr[$j]=~/diagdir/)&&!($entryarr[$j]=~/chptdir/)){
	        $spec++;
	    }
	}
	#split formula string and extract scan parameter which determines the index (first list entry)
	@parlist = split(',',$formula);
	; #check for name(X) combination where X is species/occurence index
	$string = (@parlist[0] =~/(!?[A-Z,a-z][A-Z,a-z,_,0-9]+)\((\d)\)/);
	#if string is a valid parameter with species index (N) replace start index
	if ($string ne '') {
	    #search for matching entry in scan structure
	    @parlist[0]=return_index_StrInt($1,$2);
	    if (@parlist[0] < 0) {
		print "unknown scanwith parameter $1\($2\) occured at parameter $name\($spec\)\n";
		exit(0);
	    }
	    $j=return_index_StrInt($name,$spec);
	    $StrInt->start($j,@parlist[0]); #REPLACE DUMMY STRING BY ACTUAL INDEX
	} else {
	    #now check for strings without (N) information
	    $string = (@parlist[0] =~/(!?[A-Z,a-z][A-Z,a-z,_,0-9]+)/);
	    #if not already an index, try to replace string:
	    if ($string ne '') {
		#search for matching entry in scan structure
		@parlist[0]=return_index_StrInt($1,1);
		if (@parlist[0] < 0) {
		    print "unknown scanwith parameter $1\($2\) occured at parameter $name\($spec\)\n";
		    exit(0);
		}
		$j=return_index_StrInt($name,$spec);
		$StrInt->start($j,@parlist[0]); #REPLACE DUMMY STRING BY ACTUAL INDEX
	    } #else keep start index (which then needs to be a valid index)
	}
     }
     $spec=1;
     $i++;
  }
  return();
}


sub get_noeff{
  my $parfile = shift;
  my $perf_vec = ReadValue ($parfile,"perf_vec",1,1);
  my $nblocks   = ReadValue ($parfile,'nblocks',1,0);
  my $n_procs_x = ReadValue($parfile,'n_procs_x',1,0);
  my $n_procs_y = ReadValue($parfile,'n_procs_y',1,0);
  my $n_procs_z = ReadValue($parfile,'n_procs_z',1,0);
  my $n_procs_v = ReadValue($parfile,'n_procs_v',1,0);
  my $n_procs_w = ReadValue($parfile,'n_procs_w',1,0);
  my $n_procs_s = ReadValue($parfile,'n_procs_s',1,0);
  my $n_procs_sim = ReadValue($parfile,'n_procs_sim',1,0);
  my $n_parallel_sims = ReadValue($parfile,'n_parallel_sims',1,0);
  my $res = 1;
  if ($perf_vec=~/~.*0.*!/){$res = 0;} # dont count "0"s after first "!" (fortran comment)
  if ($nblocks ==0){$res = 0;}
  if ($n_procs_x < 1){$res = 0;}
  if ($n_procs_y < 1){$res = 0;}
  if ($n_procs_z < 1){$res = 0;}
  if ($n_procs_v < 1){$res = 0;}
  if ($n_procs_w < 1){$res = 0;}
  if ($n_procs_s < 1){$res = 0;}
  if ($n_procs_sim < 1){$res = 0;}
  if ($n_parallel_sims < 1){$res = 0;}
  #only if everything is set, skip efficiency_scan/autopar!
  return $res;
}
####################################################################################



sub show_help {
    print "\nscanscript for parameter scans with the GENE code\n";
    print "-stability analysis: output eigenvalues are written to scan.log\n";
    print "-neoclassical computations: output fluxes are written to neo.log\n";
    print "\nusage: ./scanscript \n";
    print "main options: \n";
    print "--np = <int>: -total number of mpi processors, must be set correctly!\n";
    print "              -note that several instances of GENE can be started\n";
    print "              -automatic efficiency test finds optimal number of parallel GENE's\n";
    print "              (skip the test by setting the parameters n_parallel_sims and n_procs_sim)\n";
    print "--ppn =<int>: number of processors per compute node of current machine\n";
    print "              optional but recommended for automatic efficiency tests \n";
    print "--mps =<int>: maximum number of parallel simulations\n";
    print "              optional but recommended for automatic efficiency tests \n";
    print "              for low dimensional scans, a small number (~1-4) is recommended\n";
    print "--cs        : continue_scan in current diagdir, valid checkpoints are read\n";
    print "            : (changing an entry of PARINDIR/gene_status from s to f\n";
    print "            :  lets GENE ignore that run)\n";
    print "--help      : display this help text\n";
    print "--long_help : display a full list of options \n";
    print "              and more information on gene scans\n";
    print "\n";
    print "quickstart:\n";
    print "- compile gene with gmake\n";
    print "- specify scan in parameters file e.g:\n";
    print "  kymin = 0.5 !scanlist: 0.5,0.6,0.7\n";
    print "- use autoparallelization for an efficiency test (n_procs_z = -1 etc.)\n";
    print "- call (in submit script): ./scanscript --np=<int> --ppn=<int>\n";
    print "\n";
    print "for more information, see gene documentation and/or use the --long_help option\n";
}

sub show_long_help {
    print "-----------------------------------------------\n";
    print "advanced options: \n";
    print "--syscall=<string> specify GENE execution command.\n";
    print "                   default is gmake -f ../makefile run N_PES=\$N_PES\n";
    print "--eff   : test the parallel efficiency for your problem\n";
    print "          -output: efficiency.log\n";
    print "          -autoparallelization of box namelist parameters is required\n";
    print "          --np(n_pes) sets the maximum number of processes\n";
    print "--ap_switch: switch on/off using the perf_vec and MPI mapping from the first run\n";
    print "             (default: on, if box isn't changed)\n";
    print "--o='<string>' : specify the output directory\n";
    print "--test  : test settings without executing gene\n";
    print "--stop  : stop script right before gene execution\n";
    print "--noeff : suppress initial efficiency_scan\n";
    print "--force : perform all scans ,default: off (exit at first gene error)\n";
    print "--mks   : (make_scanlog) creates a scan.log of a (possibly unfinished) scan\n";
    print "          parameters file must contain original directory and scan information\n";
    print "\n-----------------------------------------------\n";
    print "how to initiate a parameter scan in the parameters file:\n";
    print "- 'range' scan: insert start-, step- and end-value:\n";
    print "   <par> = <value> !scanrange: <start>,<step>,<end>\n";
    print "   or\n";
    print "   <par> = <value> !scan: <start>,<step>,<end>\n";
    print "- 'function' scan: \n";
    print "   <par>=<value>  !scanfunc: <endstep>,<function(xi)>,<endvalue>\n";
    print "   integer variable xi [1..endstep]\n";
    print "   operators: +,-,*,/,%,e,**,abs,sqrt,int,exp,log,sin,cos.\n";
    print "   example:\n";
    print "   kymin=0 !scanfunc: 10,0.5*(xi-1)**2,4\n";
    print "- 'list' scan: \n";
    print "   examples:\n";
    print "   nz0= 1  !scanlist: 8,24,12,48\n";
    print "   collision_op = 'landau'  !scanlist: \"'none'\",\"'landau'\"\n";
    print "- 'with' scan:\n";
    print "   examples:\n";
    print "   kymin = 1 !scanlist:0.1,0.5,1\n";
    print "   kx_center = 1 !scanwith:kymin(1),0,3,-10\n";
    print "   This scans the pairs (kymin,kx_center)=(0.1,0); (0.5,3); (1,-10).\n";
    print "   The first entry after !scanwith: specifies the PRECEDING parameter to scan with \n";
    print "   (or the global index of scanned parameters in parameter file starting at 0)\n";
    print "   \n";
    print "- parameters can depend on other parameters (also in functions and lists):\n";
    print "   <par1>=0 !scan: 5*<par2>(<i>)\n";
    print "   set in brackets (<i>) the species number of par2, (1) otherwise\n";
    print "   example:\n";
    print "   omn= 2.22 !scan: 0.5*omt(1)\n";
    print "- string parameters can include values of other parameters:\n";
    print "   <par1>='file_XXXX_YYYY' !replace: <par2>(<i>),<par3>(<j>)\n";
    print "   example:\n";
    print "   geomfile= 's_alpha_XXXX_YYYY' !replace: x0(1),nz0(1)\n";
    print "-----------------------------------------------\n";
    print "scan re/im of complex variables e.g.:\n";
    print "ev_shift = 1 !scanrange:im: -1,0.2,1\n";
    print "-----------------------------------------------\n";
    #print "Initial automatic EFFICIENCY TEST determines the parameters\n";
    #print "  n_parallel_sims = number of simultanuous GENE runs\n";
    #print "  n_procs_sim = number of processors per GENE run\n";
    #print "(if not set)\n";
    #print "WARNING:\n efficiency test FAILS in cases like the following:\n";
    #print "-parallel performance is optimal for 8 processors\n";
    #print "-you scan over 4 values\n";
    #print "-you start your job on 64 processors\n";
    #print "-each problem is started with 8 processors, 32 will idle the whole time\n";
    #print " obviously it would be better to use 16 or even 32 processors per problem\n";
    #print "SOLUTION: set n_procs_sim manually!\n";
    print "\nfor more information, see gene documentation\n";
}
###############################################################################
#### COMMENTS ###########################################################################################################
#                                                                                          #
#   Traver recursion:
#   works as follows: we go down in the recursion depth to the last parameter to scan (name i+1) does not exist.
#   on the way the first values for all other parameters are written to parameters file.
#   at "last i" we go through the while loop and each time gene is executed.
#   after whilecond is negative, we go up in the recursion depth, where we are in the while loop of the "last i -1"
#   parameter. there we proceed to the next value of par. last i -1. gene is not executed, instead we go down to last i
#   again where gene can be executed. and so forth.

#   GENE call:
#   the gx($call) command executes $call and observes the STDOUT of $call
#   the scanscript reads this STDOUT and if the word ERROR or STOP occurs,
#   the scan is ended with exit(0). importand remark: the STDERR of $call is not monitored...
#                                                                                          #
#########################################################################################################################
'''