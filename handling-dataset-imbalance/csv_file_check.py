import csv

log_path = "data/self_driving_car_dataset_make/driving_log.csv"
threshold = 0.05

count_near_zero = 0
count_positive = 0
count_negative = 0
total = 0

with open(log_path, 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        try:
            angle = float(row[3])  
            total += 1

            if abs(angle) < threshold:
                count_near_zero += 1
            elif angle > 0:
                count_positive += 1
            else:
                count_negative += 1
        except:
            continue 

print(f"Total samples: {total}")
print(f"Near zero: {count_near_zero} ({(count_near_zero/total)*100:.2f}%)")
print(f"Positive angle (right turns): {count_positive} ({(count_positive/total)*100:.2f}%)")
print(f"Negative angle (left turns): {count_negative} ({(count_negative/total)*100:.2f}%)")
