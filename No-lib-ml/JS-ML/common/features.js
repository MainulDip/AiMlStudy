const features = {}

features.getPathCount = (paths) => {
    return paths.length
}

features.getPointCount = (paths) => {
    const points = paths.flat() // converting multidimensional array into single dimensional array
    return points.length
}

if (typeof module !== 'undefined') {
    module.exports = features
}