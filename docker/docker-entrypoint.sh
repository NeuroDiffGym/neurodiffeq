echo "Container is running!!!"
echo -en "\033[92m
The following commands are available:

    notebook
        Runs the jupyter notebook from http://127.0.0.1:8888/

\033[0m
"
notebook(){
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
}
export -f notebook
pipenv -v shell