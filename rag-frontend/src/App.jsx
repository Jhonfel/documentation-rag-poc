import React, { useState } from 'react';
import { Button, Input, List } from 'antd';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');

  const handleSend = async () => {
    if (inputValue.trim() !== '') {
      // Send a POST request to the /ask endpoint
      const response = await fetch('/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: inputValue })
      });

      if (response.ok) {
        const data = await response.json();
        // Append the server response to the messages list
        setMessages([...messages, `Q: ${inputValue}`, `A: ${data.response}`]);
      } else {
        const errorData = await response.json();
        setMessages([...messages, `Q: ${inputValue}`, `Error: ${errorData.error}`]);
      }
      setInputValue('');
    }
  };

  return (
    <div className="chat-container">
      <List
        header={<div>Chat Messages</div>}
        bordered
        dataSource={messages}
        renderItem={item => (
          <List.Item>{item}</List.Item>
        )}
        style={{ width: 300, height: 400, overflow: 'auto' }}
      />
      <Input.Group compact style={{ marginTop: 20 }}>
        <Input
          style={{ width: 'calc(100% - 100px)' }}
          value={inputValue}
          onChange={e => setInputValue(e.target.value)}
          onPressEnter={handleSend}
          placeholder="Enter message..."
        />
        <Button type="primary" onClick={handleSend}>Send</Button>
      </Input.Group>
    </div>
  );
}

export default App;
