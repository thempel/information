## Building docs
First, automatically create api-docs with
```bash
sphinx-apidoc --ext-autodoc -f --separate -d 2 -o source ../informant
```
second, build html with sphinx
```bash
make html
```

