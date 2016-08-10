while true; do

	for i in `ls task_queue`; do
	
		echo "Doing task " $i
		date
		./task_queue/$i > ./task_log/$i.log
		echo "Done with task " $i
		date
		mv ./task_queue/$i ./task_log/$i

	done

	echo "Finished everything"
	date
	sleep 60

done

