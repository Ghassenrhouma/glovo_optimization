with open(r'c:\Users\ghass\projet_optimisation\glovo_optimization\app\streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()
old = '""stretch""'
new = 'use_container_width=True'
content = content.replace('width=' + old, new)
with open(r'c:\Users\ghass\projet_optimisation\glovo_optimization\app\streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Fixed', content.count(new), 'occurrences')
