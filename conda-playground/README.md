
# setup 

prerequisites
```
pacman -Sy libxau libxi libxss libxtst libxcursor libxcomposite libxdamage libxfixes libxrandr libxrender mesa-libgl  alsa-lib libglvnd

```

download anaconda setup script

run it and add this to your .zsh

```
__conda_setup="$('/home/codedrift/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"

function condaenv() {
    eval "$__conda_setup"
}

```
now `condaenv` enables conda


create a new env

```
conda create --name conda-playground python=3.9

```

save env to file

```
conda env export > environment.yml

```

remove an env

```
conda remove --name conda-playground --all

```

restore an env from file

```
conda env create -f environment.yml

```


