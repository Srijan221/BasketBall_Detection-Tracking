import os
import re
# files = []
# Add the path of txt folder
for item in os.listdir("labels"):
    if item.endswith('.txt'):
    #     match = re.search("\.txt$", item)
  
    # # if match is found
    #     if match:
    #         print("The file ending with .txt is:",item)
        item = os.path.join("labels",item)
        with open(item, 'r') as myfile_read:
            lines = myfile_read.readlines()
            with open(item, 'w') as myfile_write:
                for line in lines:
                    # print(line)
                    # print("--x--")
                    # print(line.strip(" ")[4])
                    # print("---end--")
                    if (line.strip(" ")[0]=='0'):
                        print("original- ",line)
                        myfile_write.write(line)

                print(lines)
            print(item)
            print("-------")

    
    # # Decrease the first number in any line by one
    # for i in file_data:
    #     if i[0].isdigit():
    #         temp = float(i[0]) - 1
    #         i[0] = str(int(temp))

    # # Write back to the file
    # f = open(item, 'w')
    # for i in file_data:
    #     res = ""
    #     for j in i:
    #         res += j + " "
    #     f.write(res)
    #     f.write("\n")
    # f.close()