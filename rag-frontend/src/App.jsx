import React, { useState } from 'react';
import { Button, Input, List, message, Modal, Upload } from 'antd';
import { InboxOutlined } from '@ant-design/icons';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [loading, setLoading] = useState(false);  


  const handleSend = async () => {
    if (inputValue.trim() !== '') {
      setLoading(true); 
      let chatinput = inputValue
      setInputValue("")
      const response = await fetch('/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: chatinput })
        
      });
      

      if (response.ok) {
        const data = await response.json();
        setMessages([...messages, `Q: ${inputValue}`, `A: ${data.response}`]);
      } else {
        const errorData = await response.json();
        setMessages([...messages, `Q: ${inputValue}`, `Error: ${errorData.error}`]);
      }
      setLoading(false);  

    }
  };

  const uploadProps = {
    name: 'file',
    multiple: false,
    action: '/upload',
    onChange(info) {
      const { status } = info.file;
      if (status === 'done') {
        message.success(`${info.file.name} file uploaded successfully.`);
      } else if (status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
    onDrop(e) {
      console.log('Dropped files', e.dataTransfer.files);
    },
  };

  const fetchAndUploadDocumentation = async () => {
    // Assuming the endpoint '/fetch-docs' will handle fetching from Amazon docs and uploading
    const response = await fetch('/fetch-docs');
    if (response.ok) {
      message.success('Documentation fetched and uploaded successfully.');
    } else {
      message.error('Failed to fetch and upload documentation.');
    }
  };

  const showUploadModal = () => {
    Modal.confirm({
      title: 'Upload Knowledge Base',
      content: (
        <div>
          <Upload.Dragger {...uploadProps}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">Click or drag file to this area to upload</p>
            <p className="ant-upload-hint">
              Support for a single file upload. Strictly prohibit from uploading sensitive data.
            </p>
          </Upload.Dragger>
          {/*<Button type="primary" onClick={fetchAndUploadDocumentation} style={{ marginTop: 16 }}>
            Use Current Web Documentation
          </Button>*/}
        </div>
      ),
      okText: 'Close',
      onOk() {},
    });
  };

  return (
    <div className="chat-container">
      <Button type="primary" onClick={showUploadModal}>
        Update Knowledge Base
      </Button>
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
        <Button type="primary" onClick={handleSend} loading={loading}>Send</Button>
      </Input.Group>
    </div>
  );
}

export default App;
