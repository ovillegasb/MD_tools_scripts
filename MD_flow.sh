#!/bin/bash

#SBATCH --job-name=MDFLOW
#SBATCH --workdir=.
#SBATCH --partition=short
#SBATCH --account=uppa
#SBATCH --time=0-12:0:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2000

# dossier de travail (securite)
module purge
module load gromacs/2019.3
module load gromacs/2019.3-seq
module list

# COMMANDS
echo "------------------------------------------------------------"
type mpiexec
echo "User:" $USER
echo "Date:" `date`
echo "Host:" `hostname`
echo "Directory:" `pwd`
echo "SCRATCHDIR           =  $SCRATCHDIR"
echo "SLURM_JOB_NAME       =  $SLURM_JOB_NAME"
echo "SLURM_JOB_ID         =  $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST   =  $SLURM_JOB_NODELIST"
echo ""
echo "SLURM_JOB_NUM_NODES  =  $SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS         =  $SLURM_NTASKS"
echo "SLURM_TASKS_PER_NODE =  $SLURM_TASKS_PER_NODE"
echo "SLURM_CPUS_PER_TASK  =  $SLURM_CPUS_PER_TASK"
lscpu | grep CPU\(s\):
echo "------------------------------------------------------------"

# EXECUTION in batch mode with the mpiexec command

# Set OMP_NUM_THREADS to SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export GMX_MAXCONSTRWARN=-1

# works variables
EXP=exp.asp.psol
HW=~/${EXP}
MDP=${HW}/mdp
SCRIPTS=${HW}/scripts
MOL=${HW}/molecules

# MINIMIZATION
if [ ! -f min/confout.gro ]
then
    rm -vfR min
    mkdir min
    gmx grompp -f $MDP/em.mdp -c box.gro -p topol.top -o min/run.tpr -po min/mdout.mdp
    cd min
    mpiexec -n $SLURM_NTASKS gmx_mpi mdrun -s run.tpr -v -cpi state.cpt -cpt 2
    cd ../
else
    echo "MINIMIZATION complet"
fi

# EQUILIBRATION NVT
if [ ! -f nvt/confout.gro ] || [ -f min/confout.gro ]
then
    if [ -f nvt/state.cpt ]
    then
        echo "EQ NVT not complete"
    else
        rm -vfR nvt
        mkdir nvt
        gmx grompp -f $MDP/nvt.mdp -c min/confout.gro -p topol.top -o nvt/run.tpr -po nvt/mdout.mdp
    fi

    cd nvt
    mpiexec -n $SLURM_NTASKS gmx_mpi mdrun -s run.tpr -v -cpi state.cpt -cpt 2 -nstlist 5
    cd ..

else
    echo "EQ NVT complet"
fi

# EQUILIBRATION NPT
if [ ! -f npt/confout.gro ] || [ -f nvt/state.cpt ]
then
    if [ -f npt/state.cpt ]
    then
        echo "EQ NPT not complete"
    else
        rm -vfR npt
        mkdir npt
        gmx grompp -f $MDP/npt.mdp -c nvt/confout.gro -t nvt/state.cpt -p topol.top -o npt/run.tpr -po npt/mdout.mdp
    fi

    cd npt
    mpiexec -n $SLURM_NTASKS gmx_mpi mdrun -s run.tpr -v -cpi state.cpt -cpt 2 -nstlist 5
    cd ..

else
    echo "EQ NPT complet"
fi

# PRODUCTION MD
if [ ! -f md/confout.gro ] || [ -f npt/state.cpt ]
then
    if [ -f md/state.cpt ]
    then
        echo "MD not complete"
    else
        rm -vfR md
        mkdir md
        gmx grompp -f $MDP/md.mdp -c npt/confout.gro -t npt/state.cpt -p topol.top -o md/run.tpr -po md/mdout.mdp -maxwarn 1
    fi

    cd md
    mpiexec -n $SLURM_NTASKS gmx_mpi mdrun -s run.tpr -v -cpi state.cpt -cpt 2 -nstlist 5
    cd ..

else
    echo "MD complet"
fi
