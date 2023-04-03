import os

for item in os.listdir("labels"):
    if item.endswith('.txt'):
        item = os.path.join("labels",item)
        with open(item, 'r') as myfile_read:
            lines = myfile_read.readlines()
            with open(item, 'w') as myfile_write:
                for line in lines:
                    if (line.strip(" ")[0]=='0'):
                        print("original- ",line)
                        myfile_write.write(line)

                print(lines)
            print(item)
            print("-------")
