class Point {
  constructor(x, y) {
    this.x = x;
    this.y = y;
  }

  dist(pt) {
    return sqrt((pt.x-this.x)**2 + (pt.y-this.y)**2)
  }
}
