### Collected Data Overview:
The collected data (images) are inside raw directory in json format. Each json file contain 8 drawing inside  and some other identifiers.
```js
{
    session : 1234,
    student : "Name",
    drawings : [
        car : [
                [ [x,y], [x,y], [x,y], ... ] // path 0, get the line by joining them (x,y cordinate of mousedown and move) 
                [ [x,y], [x,y], [x,y], ... ], // path 1
                ...
            ],
        fish : ... ,
        house : ... ,
        tree : ... ,
        bicycle : ... ,
        guitar : ... ,
        pencil : ... ,
        clock : ...
    ]
}
```