


class A(object):

    def __init__(self, msg):
        self.msg = msg

    def printout(self):
        print self.msg


class B(object):

    def __init__(self, printer, msg):
        self.printer = printer
        self.msg = msg

    def print_own(self):
        print self.msg

    def printout(self):
        self.printer()


a = A('A')
print 'a',
a.printout()

b = B(a.printout, 'B')
print 'b'
print 'own',
b.print_own()
print 'printout',
b.printout()
