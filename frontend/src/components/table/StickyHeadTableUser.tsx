import * as React from 'react';
import Paper from '@mui/material/Paper';
import Table from '@mui/material/Table';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableContainer from '@mui/material/TableContainer';
import TableHead from '@mui/material/TableHead';
import TableRow from '@mui/material/TableRow';
import { MdDelete } from "react-icons/md";
import TextField from '@mui/material/TextField';
import {
  getUsers,
  deleteUser,
} from '../../helpers/api-admin/user';
import { useEffect, useState } from 'react';

interface Column {
  id: 'name' | 'email' | 'deleteID';
  label: string;
  minWidth?: number;
  align?: 'left' | 'center';
  format?: (value: number) => string;
}

const columns: readonly Column[] = [
  { id: 'name', label: 'Name', minWidth: 50 },
  { id: 'email', label: 'Email', minWidth: 100 },
  {
    id: 'deleteID',
    label: 'Action',
    minWidth: 100,
    align: 'center',
  },
];

interface Data {
  name: string;
  email: string;
  deleteID: string;
}

function createData(
  name: string,
  email: string,
  deleteID: string
): Data {
  return { name, email, deleteID };
}

export default function StickyHeadTableUser() {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [rows, setRows] = useState<Data[]>([]);

  useEffect(() => {
    async function fetchData() {
    try {
      const data = await getUsers();
      const updatedRows = data.map((item: { [x: string]: string }) =>
        createData(
          item['name'],
          item['email'],
          item['_id']
        )
      );
      updatedRows.push(createData('', '', ''));
      
      setRows(updatedRows);
    } catch (error) {
      // Handle the exception
      console.error('An error occurred:', error);
      
      // Create fake data
      const fakeData = createData('Fake Name', 'Fake Email', 'fakeId');
      const updatedRows = [fakeData];
      
      setRows(updatedRows);
    }
  }

    fetchData();
  }, []);

  const handleDelete = async (id: string) => {
    try {
      await deleteUser(id);
      const updatedRows = rows.filter((row) => row.deleteID !== id);
      setRows(updatedRows);
    } catch (error) {
      console.error('Error deleting user:', error);
    }
  };
  


  return (
    <Paper sx={{ width: '100%', overflow: 'hidden' }}>
      <TableContainer sx={{ maxHeight: 440 }}>
        <Table stickyHeader aria-label="sticky table" sx={{ background: 'rgba(0, 0, 0, 0.7)', color: 'white' }}>
          <TableHead>
            <TableRow>
              {columns.map((column) => (
                <TableCell
                  key={column.id}
                  align={column.align}
                  style={{
                    color: 'white',
                    background: 'gray',
                    fontWeight: 'bold',
                    fontSize: '1.2rem',
                    flex: 1, // Set equal flex for each column
                    wordWrap:'break-word', // Enable content wrapping
                    overflow: 'hidden', // Hide overflowing text
                    textOverflow: 'ellipsis', // Display ellipsis for overflowing text
                  }}
                >
                  {column.label}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {rows
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((row, index) => {
                const isLastRow = index === rows.length - 1;
                return (
                  <TableRow hover role="checkbox" tabIndex={-1} key={row.email}>
                    {columns.map((column) => {
                      const value = row[column.id];
                      return (
                        <TableCell
                          key={column.id}
                          align={column.align}
                          style={{
                            color: 'white',
                            flex: 1, // Set equal flex for each column
                            wordWrap: 'break-word', // Enable content wrapping
                            overflow: 'hidden', // Hide overflowing text
                            textOverflow: 'ellipsis', // Display ellipsis for overflowing text
                          }}
                        >
                          {column.id === 'deleteID' ? (
                              isLastRow ?
                                (
                              <button
                                style={{
                                  color: 'white',
                                  background: 'green',
                                  border: 'none',
                                  padding: '5px 10px',
                                  borderRadius: '5px',
                                  cursor: 'pointer',
                                  fontWeight: '700',
                                  width: '100%'
                                }}
                              >
                                Add
                              </button>
                              ) : (
                              <div style={{
                                display: 'flex',
                                alignItems: 'center', 
                                justifyContent: 'center'
                              }}>
                                  <MdDelete style={{ fontSize: "24px", cursor: "pointer" }} 
                                    onClick={() => handleDelete(value.toString())}
                                  />
                              </div>
                            )
                          ) : (
                            <input
                              type="text"
                              value={value.toString()}
                              contentEditable="true"
                              style={{
                                color: 'white',
                                background: "rgba(0, 0, 0, 0)",
                                border: 'none',
                                flex: 1, // Set equal flex for each column
                                wordWrap: 'break-word', // Enable content wrapping
                                overflow: 'hidden', // Hide overflowing text
                                textOverflow: 'ellipsis', // Display ellipsis for overflowing text
                              }}
                                />
                            // <TextField value={value.toString()} id="outlined-basic" style={{
                            //   color: 'white',
                            // }}/>
                          )}
                        </TableCell>
                      );
                    })}
                  </TableRow>
                );
              })}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
}