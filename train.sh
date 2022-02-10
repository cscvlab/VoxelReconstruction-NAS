run(){
  python run.py \
  --vox_path $1
}

for file in ./DATASETS/*
do
  if test -f $file && !(test -e "./LOGS/"$(basename $file .obj))
  then
      echo $file
      run $file
  else
      echo $file" already trained"
  fi
done
