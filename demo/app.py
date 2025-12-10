from flask import Flask, request, jsonify
import sqlite3
from .waf import waf
app = Flask(__name__)

def init_db(): #init db (using sqlite)
    conn = sqlite3.connect('demo.db')
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS users') # delete table on first exec
    c.execute('CREATE TABLE users (id text, pw text)')
    c.execute("INSERT INTO users VALUES ('testuser','5dccc7bf2d9d9897c411e7d4b1b99480')")
    conn.commit()
    conn.close()

#both routes /secure,/vulnerable are using the same query construction method

@app.route('/secure') # waf-protected route
@waf # waf decorator
def secure():
    try:
        id = request.args.get('id', '')
        pw = request.args.get('pw', '')
        
        conn = sqlite3.connect('demo.db')
        c = conn.cursor()
        
        query = f"SELECT * FROM users WHERE id = '{id}' and pw = '{pw}'"
        c.execute(query)
        result = c.fetchall()
        conn.close()
        
        if result:
            return f"login success! your id: {result[0][0]}, your pw: {result[0][1]}"
        return jsonify({'status': 'login failed'})
    except:
        return jsonify({'status': 'internal server error'})
    
@app.route('/vulnerable') # unprotected route
def vulnerable():
    try:
        id = request.args.get('id', '')
        pw = request.args.get('pw', '')
        
        conn = sqlite3.connect('demo.db')
        c = conn.cursor()
        
        query = f"SELECT * FROM users WHERE id = '{id}' and pw = '{pw}'"
        c.execute(query)
        result = c.fetchall()
        conn.close()
        
        if result:
            return f"login success! your id: {result[0][0]}, your pw: {result[0][1]}"
        return jsonify({'status': 'login failed'})
    except:
        return jsonify({'status': 'internal server error'})

@app.route('/') # index page
def index():
    return '''
    <h1>Demo</h1>
    <h2>vulnerable</h2>
    <form action="/vulnerable">
    <input name="id" placeholder="id"/>
    <input name="pw" placeholder="pw"/>
    <button type="submit">login</button>
    </form>

    <h2>secure</h2>
    <form action="/secure">
    <input name="id" placeholder="id"/>
    <input name="pw" placeholder="pw"/>
    <button type="submit">login</button>
    </form>
    '''

if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)