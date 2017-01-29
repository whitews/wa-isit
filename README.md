# wa-isit
Web Accessible Image Sub-region Identifier Tool

![Screenshot of wa-isit](screenshot.png)

## Staning up with docker
```
docker build -t trainor .

docker run -d -p 80:80 \
-v $(pwd):/trainor \
--name trainor \
--restart=always \
trainor /trainor/standup.sh
```   

## Live Demo
http://rapid-235.vm.duke.edu
