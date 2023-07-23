const features = {}

features.getPathCount = (path) => {
    return paths.length
}

features.getPointcount = (paths) => {
    const points = paths.flat() // converting multidimensional array into single dimensional array
    return points.length
}

if (typeof module !== 'undefined') {
    module.exports = features
}