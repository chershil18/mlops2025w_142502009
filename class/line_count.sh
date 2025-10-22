exec 10<&0
exec < $1
in=$1
file="current_line.txt"
let count=0

while read LINE
do
	((count++))
	echo $LINE > $file
	if [ $? -ne 0 ]
	then echo "Error in  writing to file ${file}; check its permissions!"
	fi
done
echo "Number of lines: $count"
echo "The last line of the file is: `cat ${file}`"
echo "EXpected number of lines: `wc -l $in`"

exec 0<&10 10<&-
