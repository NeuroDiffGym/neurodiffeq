echo "Container is running!!!"
echo -en "\033[92m
The following commands are available:

    notebook
        Runs the jupyter notebook from http://127.0.0.1:9898/

\033[0m"
notebook (){
    jupyter notebook --ip 0.0.0.0 --port 9898 --no-browser --allow-root
}

export -f notebook
pipenv -v shell