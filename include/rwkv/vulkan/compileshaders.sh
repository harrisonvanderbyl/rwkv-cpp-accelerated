# enumerate all files in ops/** with .comp extension and compile them to .spv
files=$(ls ops/**/*.comp)
filestodelete=$(ls **/*.spv)

for file in $filestodelete; do
    echo "Deleting $file"
    rm $file
done

for file in $files; do
    echo "Compiling $file"
    # get only the filename
    filename=$(basename -- "$file")
    glslc $file -o ./${filename%.*}.spv
done 

