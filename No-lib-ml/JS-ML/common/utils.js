const utils = {}

utils.flaggedUsers=
   [1663882102141,1663900040545,1664485938220]; // drew nsfw
// using later
utils.styles={
    car:{color:'gray',text:'🚗'},
    fish:{color:'red',text:'🐠'},
    house:{color:'yellow',text:'🏠'},
    tree:{color:'green',text:'🌳'},
    bicycle:{color:'cyan',text:'🚲'},
    guitar:{color:'blue',text:'🎸'},
    pencil:{color:'magenta',text:'✏️'},
    clock:{color:'lightgray',text:'🕒'},
 };

utils.printProgress = (count, max) => {
    process.stdout.clearLine()
    process.stdout.cursorTo(0)
    const percent = utils.formatPercent(count/max)
    process.stdout.write(count + "/" + max + " (" + percent + ")")
}

utils.formatPercent = (n) => {
    return (n*100).toFixed(2) + "%"
}

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

if (typeof module!=='undefined') {
    module.exports = utils
}