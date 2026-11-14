from ai_generator import generate_site_structure

structure = generate_site_structure("Барбершоп в Казани, мужские стрижки, запись онлайн")
print(structure.model_dump_json(indent=2, exclude_none=True))