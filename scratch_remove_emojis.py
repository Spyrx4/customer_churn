import os
import re
import codecs

for root, dirs, files in os.walk('views'):
    for file in files:
        if file.endswith('.py'):
            filepath = os.path.join(root, file)
            with codecs.open(filepath, 'r', 'utf-8') as f:
                content = f.read()
            
            # Replace common remnants
            new_content = content.replace('" ', '"')
            new_content = new_content.replace('###  ', '### ')
            new_content = new_content.replace('##  ', '## ')
            
            if content != new_content:
                with codecs.open(filepath, 'w', 'utf-8') as f:
                    f.write(new_content)
                print(f"Cleaned up spaces in {filepath}")
