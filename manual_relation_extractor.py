# This manual relation extractor finds out the institution relation based on regex patterns.
# It looks for the regular expressions:
# 1) educated at
# 2) graduated from
# 3) matriculated at
# 4) attended
# 5) studied at

import re

if __name__  == "__main__":
    strs = "Wassup? Are you okay? Not Ok"

    match = re.search(r'\bNot Ok\b', strs)
    if match:
        print "Found"
    else:
        print "Not Found"