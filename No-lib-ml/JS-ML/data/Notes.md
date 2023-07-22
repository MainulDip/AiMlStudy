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

### Grouping Large Data:
```js
// GroupBy Sorting. Like send the object array and label as key
utils.groupBy = (objArray, key) => {
    const groups = {}
    for ( let obj of objArray ) {
        
        const val = obj[key] // getting value by key, ie: label or student_id session number
        
        // assigning empty array when groups[key] is initialized as object property
        if (groups[val] == null ) {
            groups[val] = []
        }

        // then push value to the array for matching iteration 
        groups[val].push(obj)
        /**
         * Signature: groups = { 1234: [{},{}], 2345: [{},{}], ...}
         */
    }

    return groups
}